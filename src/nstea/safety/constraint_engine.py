"""Symbolic Constraint Engine — formal validator using the clinical knowledge graph.

Replaces/extends the YAML rule engine with graph-based constraint validation.
Queries the knowledge graph to check proposed drugs against patient data,
producing structured constraint violations.

Key principle: "Honest about incompleteness" — unmapped actions get flagged
for mandatory human-in-the-loop review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from nstea.models.patient import PatientInput
from nstea.safety.knowledge_graph import ClinicalKnowledgeGraph, build_default_knowledge_graph

logger = structlog.stdlib.get_logger(__name__)

# Module-level singleton
_knowledge_graph: ClinicalKnowledgeGraph | None = None


def _get_knowledge_graph() -> ClinicalKnowledgeGraph:
    """Get or create the default knowledge graph singleton."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = build_default_knowledge_graph()
    return _knowledge_graph


@dataclass
class ConstraintViolation:
    """A single constraint violation found during validation."""

    rule: str  # What constraint was violated
    severity: Literal["info", "warning", "critical"]
    description: str  # Human-readable description
    action_blocked: str  # What was prevented
    source: str = ""  # Knowledge graph source (guideline, interaction DB, etc.)
    alternative: str = ""  # Suggested alternative if available


@dataclass
class ConstraintResult:
    """Full result from symbolic constraint validation."""

    is_safe: bool
    violations: list[ConstraintViolation] = field(default_factory=list)
    unmapped_actions: list[str] = field(default_factory=list)
    guideline_alignment: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(v.severity == "critical" for v in self.violations)

    @property
    def requires_human_review(self) -> bool:
        """Human review is mandatory if there are unmapped actions or critical violations."""
        return bool(self.unmapped_actions) or self.has_critical

    @property
    def review_reason(self) -> str:
        reasons = []
        if self.has_critical:
            reasons.append(f"{sum(1 for v in self.violations if v.severity == 'critical')} critical constraint violation(s)")
        if self.unmapped_actions:
            reasons.append(f"{len(self.unmapped_actions)} unmapped action(s) not in knowledge graph")
        return "; ".join(reasons) if reasons else ""


class SymbolicConstraintEngine:
    """Knowledge-graph-backed formal constraint validator.

    Checks proposed actions against:
    1. Drug-drug interactions
    2. Drug-condition contraindications
    3. Allergy cross-reactivities
    4. Guideline alignment
    5. Unmapped actions (honest incompleteness)
    """

    def __init__(self, kg: ClinicalKnowledgeGraph | None = None):
        self.kg = kg or _get_knowledge_graph()

    def validate(
        self,
        proposed_drugs: list[str],
        patient: PatientInput,
    ) -> ConstraintResult:
        """Validate proposed drugs against patient data using the knowledge graph.

        Args:
            proposed_drugs: List of drug names being proposed.
            patient: Patient data with conditions, medications, allergies.

        Returns:
            ConstraintResult with violations, unmapped actions, and guideline alignment.
        """
        violations: list[ConstraintViolation] = []
        unmapped: list[str] = []
        guideline_alignment: list[dict[str, Any]] = []

        current_meds = [m.name for m in patient.medications]
        allergy_names = [a.substance for a in patient.allergies]
        condition_names = [c.name for c in patient.conditions]

        for drug in proposed_drugs:
            # Check if drug is in knowledge graph
            if not self.kg.is_mapped(drug):
                unmapped.append(drug)
                logger.info("unmapped_drug", drug=drug)
                continue

            # 1. Check allergy direct match
            for allergy in allergy_names:
                if self._names_match(drug, allergy):
                    violations.append(ConstraintViolation(
                        rule="ALLERGY_DIRECT_MATCH",
                        severity="critical",
                        description=f"'{drug}' directly matches patient allergy '{allergy}'",
                        action_blocked=drug,
                        source="Patient allergy list",
                    ))

            # 2. Check allergy cross-reactivities
            for allergy in allergy_names:
                cross_reactions = self.kg.get_allergy_cross_reactions(allergy)
                for reaction in cross_reactions:
                    if self._names_match(drug, reaction["drug"]):
                        violations.append(ConstraintViolation(
                            rule="ALLERGY_CROSS_REACTIVITY",
                            severity="critical" if reaction.get("risk_level") == "high" else "warning",
                            description=f"'{drug}' cross-reacts with patient allergy '{allergy}' (risk: {reaction.get('risk_level', 'unknown')})",
                            action_blocked=drug,
                            source="Knowledge graph: allergy cross-reactivity",
                            alternative=reaction.get("alternative", ""),
                        ))

            # 3. Check drug-condition contraindications
            contraindications = self.kg.get_contraindications(drug)
            for ci in contraindications:
                for condition in condition_names:
                    if self._names_match(condition, ci["condition"]):
                        violations.append(ConstraintViolation(
                            rule="CONDITION_CONTRAINDICATION",
                            severity=ci.get("severity", "critical"),
                            description=f"'{drug}' is contraindicated in '{condition}': {ci.get('reason', '')}",
                            action_blocked=drug,
                            source="Knowledge graph: contraindication",
                        ))

            # 4. Check drug-drug interactions (with current + proposed)
            all_meds = current_meds + [d for d in proposed_drugs if d != drug]
            interactions = self.kg.get_drug_interactions(drug)
            for interaction in interactions:
                for med in all_meds:
                    if self._names_match(med, interaction["drug"]):
                        violations.append(ConstraintViolation(
                            rule="DRUG_INTERACTION",
                            severity=interaction.get("severity", "warning"),
                            description=f"Interaction: '{drug}' + '{med}' → {interaction.get('effect', 'unknown effect')}",
                            action_blocked=drug,
                            source="Knowledge graph: drug interaction",
                        ))

        # 5. Check guideline alignment for patient conditions
        for condition in condition_names:
            recs = self.kg.get_guideline_recommendations(condition)
            for rec in recs:
                is_proposed = any(self._names_match(d, rec["drug"]) for d in proposed_drugs)
                guideline_alignment.append({
                    "condition": condition,
                    "recommended_drug": rec["drug"],
                    "guideline": rec["guideline"],
                    "strength": rec["strength"],
                    "is_proposed": is_proposed,
                })

        # Deduplicate violations by (rule, action_blocked)
        seen = set()
        unique_violations = []
        for v in violations:
            key = (v.rule, v.action_blocked.lower())
            if key not in seen:
                seen.add(key)
                unique_violations.append(v)

        is_safe = not any(v.severity == "critical" for v in unique_violations)

        result = ConstraintResult(
            is_safe=is_safe,
            violations=unique_violations,
            unmapped_actions=unmapped,
            guideline_alignment=guideline_alignment,
        )

        logger.info(
            "constraint_validation_complete",
            proposed_count=len(proposed_drugs),
            violations=len(unique_violations),
            unmapped=len(unmapped),
            is_safe=is_safe,
        )

        return result

    @staticmethod
    def _names_match(a: str, b: str) -> bool:
        """Case-insensitive fuzzy match."""
        a_lower = a.lower().strip()
        b_lower = b.lower().strip()
        if not a_lower or not b_lower:
            return False
        return a_lower in b_lower or b_lower in a_lower
