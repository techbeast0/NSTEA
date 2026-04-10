"""Hard-coded clinical safety rule engine for NS-TEA Phase 1.

Loads contraindication and drug-interaction rules from YAML and evaluates
them against patient data and proposed recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from nstea.config import settings
from nstea.models.patient import PatientInput


@dataclass
class Violation:
    """A single rule violation found during safety checking."""

    rule_id: str
    severity: Literal["info", "warning", "critical"]
    message: str
    blocked_drug: str | None = None
    related_rule_type: str = ""


@dataclass
class SafetyResult:
    """Aggregated result from running all rules against a case."""

    is_safe: bool
    violations: list[Violation] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(v.severity == "critical" for v in self.violations)


class RuleEngine:
    """YAML-driven clinical safety rule engine."""

    def __init__(self, rules_dir: Path | None = None):
        self.rules_dir = rules_dir or (settings.project_root / "data" / "rules")
        self.contraindication_rules: list[dict] = []
        self.interaction_rules: list[dict] = []
        self._load_rules()

    def _load_rules(self) -> None:
        contra_path = self.rules_dir / "contraindications.yml"
        if contra_path.exists():
            with open(contra_path) as f:
                data = yaml.safe_load(f)
            self.contraindication_rules = data.get("rules", [])

        interaction_path = self.rules_dir / "drug_interactions.yml"
        if interaction_path.exists():
            with open(interaction_path) as f:
                data = yaml.safe_load(f)
            self.interaction_rules = data.get("interactions", [])

    def check_proposed_drugs(
        self, patient: PatientInput, proposed_drugs: list[str]
    ) -> SafetyResult:
        """Check proposed drugs against patient allergies, conditions, and current meds.

        Args:
            patient: The patient data.
            proposed_drugs: List of drug names being proposed in recommendations.

        Returns:
            SafetyResult with any violations found.
        """
        violations: list[Violation] = []

        # Normalize once
        allergy_names = [a.substance.lower() for a in patient.allergies]
        condition_names = [c.name.lower() for c in patient.conditions]
        current_meds = [m.name.lower() for m in patient.medications]
        proposed_lower = [d.lower() for d in proposed_drugs]

        # 1. Check contraindication rules
        for rule in self.contraindication_rules:
            rule_type = rule.get("type", "")

            if rule_type == "allergy_match":
                # Direct allergy-drug match
                for drug in proposed_lower:
                    for allergy in allergy_names:
                        if _fuzzy_match(drug, allergy):
                            violations.append(Violation(
                                rule_id=rule["id"],
                                severity="critical",
                                message=f"ALLERGY: '{drug}' matches patient allergy '{allergy}'",
                                blocked_drug=drug,
                                related_rule_type="allergy_match",
                            ))

            elif rule_type == "cross_reactivity":
                triggers = [t.lower() for t in rule.get("trigger_allergy", [])]
                if any(_fuzzy_match(a, t) for a in allergy_names for t in triggers):
                    # Patient has the trigger allergy — check drugs
                    blocked = [d.lower() for d in rule.get("contraindicated_drugs", [])]
                    warned = [d.lower() for d in rule.get("warn_drugs", [])]

                    for drug in proposed_lower:
                        if any(_fuzzy_match(drug, b) for b in blocked):
                            violations.append(Violation(
                                rule_id=rule["id"],
                                severity="critical",
                                message=rule.get("message", f"Cross-reactivity: {drug} contraindicated"),
                                blocked_drug=drug,
                                related_rule_type="cross_reactivity",
                            ))
                        elif any(_fuzzy_match(drug, w) for w in warned):
                            violations.append(Violation(
                                rule_id=rule["id"],
                                severity="warning",
                                message=f"Cross-reactivity caution: '{drug}' may cross-react. {rule.get('message', '')}",
                                blocked_drug=drug,
                                related_rule_type="cross_reactivity",
                            ))

            elif rule_type == "condition_drug":
                triggers = [t.lower() for t in rule.get("trigger_condition", [])]
                if any(_fuzzy_match(c, t) for c in condition_names for t in triggers):
                    blocked = [d.lower() for d in rule.get("contraindicated_drugs", [])]
                    warned = [d.lower() for d in rule.get("warn_drugs", [])]

                    for drug in proposed_lower:
                        if any(_fuzzy_match(drug, b) for b in blocked):
                            violations.append(Violation(
                                rule_id=rule["id"],
                                severity=rule.get("severity", "critical"),
                                message=rule.get("message", f"Condition contraindication: {drug}"),
                                blocked_drug=drug,
                                related_rule_type="condition_drug",
                            ))
                        elif any(_fuzzy_match(drug, w) for w in warned):
                            violations.append(Violation(
                                rule_id=rule["id"],
                                severity="warning",
                                message=f"Caution with '{drug}': {rule.get('message', '')}",
                                blocked_drug=drug,
                                related_rule_type="condition_drug",
                            ))

        # 2. Check drug-drug interactions (current meds + proposed)
        all_drugs = current_meds + proposed_lower
        for rule in self.interaction_rules:
            drug_a_list = [d.lower() for d in rule.get("drug_a", [])]
            drug_b_list = [d.lower() for d in rule.get("drug_b", [])]

            a_present = [d for d in all_drugs if any(_fuzzy_match(d, a) for a in drug_a_list)]
            b_present = [d for d in all_drugs if any(_fuzzy_match(d, b) for b in drug_b_list)]

            if a_present and b_present:
                # Only flag if at least one of the interacting drugs is newly proposed
                newly_proposed_involved = (
                    any(d in proposed_lower for d in a_present)
                    or any(d in proposed_lower for d in b_present)
                )
                if newly_proposed_involved:
                    violations.append(Violation(
                        rule_id=rule["id"],
                        severity=rule.get("severity", "warning"),
                        message=(
                            f"INTERACTION: {a_present[0]} + {b_present[0]} — "
                            f"{rule.get('effect', '')}. {rule.get('recommendation', '')}"
                        ),
                        blocked_drug=None,
                        related_rule_type="drug_interaction",
                    ))

        is_safe = not any(v.severity == "critical" for v in violations)
        return SafetyResult(is_safe=is_safe, violations=violations)


def _fuzzy_match(text: str, target: str) -> bool:
    """Simple fuzzy matching: checks if target is a substring of text or vice versa."""
    text = text.lower().strip()
    target = target.lower().strip()
    if not text or not target:
        return False
    return target in text or text in target
