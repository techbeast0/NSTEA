"""Generalized evaluation harness for NS-TEA (Phase 2).

Supports multiple evaluation suites (safety, accuracy, latency) with
configurable datasets, metrics, and report generation.

Usage:
    python scripts/run_evaluation_v2.py                    # Run all suites
    python scripts/run_evaluation_v2.py --suite safety     # Run one suite
    python scripts/run_evaluation_v2.py --report json --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nstea.agents.orchestrator import run_pipeline_async
from nstea.models.patient import PatientInput


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    passed: bool
    elapsed_s: float
    metrics: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class SuiteResult:
    suite_name: str
    total: int
    passed: int
    failed: int
    elapsed_s: float
    case_results: list[CaseResult] = field(default_factory=list)
    aggregate_metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

async def evaluate_safety_case(tc: dict) -> CaseResult:
    """Evaluate a single safety edge case."""
    case_id = tc["case_id"]
    expected = tc["expected"]
    patient = PatientInput(**tc["patient"])

    start = time.time()
    try:
        analysis = await run_pipeline_async(patient)
    except Exception as e:
        return CaseResult(case_id=case_id, passed=False, elapsed_s=time.time() - start, error=str(e))
    elapsed = time.time() - start

    all_recs_text = " ".join(r.action.lower() for r in analysis.recommendations)
    all_flags_text = " ".join(f.message.lower() for f in analysis.safety_flags)
    raw_lower = (analysis.raw_llm_output or "").lower()

    # Safety — banned substances not in recommendations
    violations = [b for b in expected["must_not_recommend"] if b.lower() in all_recs_text]
    safety_pass = len(violations) == 0

    # Flag detection
    total_flags = expected.get("should_flag_critical", [])
    flagged = [f for f in total_flags if f.lower() in all_flags_text or f.lower() in raw_lower]
    flag_score = len(flagged) / len(total_flags) if total_flags else 1.0

    # Alternative suggestions
    total_alts = expected.get("should_recommend_alternative", [])
    found_alts = [a for a in total_alts if a.lower() in all_recs_text or a.lower() in raw_lower]
    alt_score = len(found_alts) / len(total_alts) if total_alts else 1.0

    # Parse success
    parse_ok = (
        "FREE-TEXT" not in analysis.diagnosis.primary
        and "VALIDATION ERROR" not in analysis.diagnosis.primary
    )

    human_review_correct = analysis.requires_human_review == expected.get("requires_human_review", True)
    overall_pass = safety_pass and parse_ok

    return CaseResult(
        case_id=case_id,
        passed=overall_pass,
        elapsed_s=round(elapsed, 1),
        metrics={
            "safety_no_violations": safety_pass,
            "parse_success": parse_ok,
            "flag_detection_score": round(flag_score, 2),
            "alt_suggestion_score": round(alt_score, 2),
            "human_review_correct": human_review_correct,
            "confidence": round(analysis.confidence.overall, 2),
            "num_safety_flags": len(analysis.safety_flags),
            "num_critical_flags": sum(1 for f in analysis.safety_flags if f.level == "critical"),
            "violations": violations,
        },
    )


async def evaluate_latency_case(tc: dict) -> CaseResult:
    """Evaluate a single case for latency measurement."""
    case_id = tc["case_id"]
    patient = PatientInput(**tc["patient"])
    target_s = tc.get("target_latency_s", 30)

    start = time.time()
    try:
        analysis = await run_pipeline_async(patient)
    except Exception as e:
        return CaseResult(case_id=case_id, passed=False, elapsed_s=time.time() - start, error=str(e))
    elapsed = time.time() - start

    parse_ok = (
        "FREE-TEXT" not in analysis.diagnosis.primary
        and "VALIDATION ERROR" not in analysis.diagnosis.primary
    )

    return CaseResult(
        case_id=case_id,
        passed=elapsed <= target_s and parse_ok,
        elapsed_s=round(elapsed, 1),
        metrics={
            "target_s": target_s,
            "within_target": elapsed <= target_s,
            "parse_success": parse_ok,
        },
    )


# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------

SUITES = {
    "safety": {
        "dataset": "data/test_cases/safety_edge_cases.json",
        "evaluator": evaluate_safety_case,
        "description": "Safety edge-case evaluation (banned drugs, flag detection, alternatives)",
    },
    # Future suites:
    # "medqa": { "dataset": "data/eval/medqa_sample.json", "evaluator": evaluate_medqa_case },
    # "latency": { "dataset": "data/eval/latency_cases.json", "evaluator": evaluate_latency_case },
}


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

async def run_suite(suite_name: str, verbose: bool = True) -> SuiteResult:
    """Run a single evaluation suite."""
    suite_cfg = SUITES[suite_name]
    dataset_path = PROJECT_ROOT / suite_cfg["dataset"]
    evaluator = suite_cfg["evaluator"]

    with open(dataset_path) as f:
        test_cases = json.load(f)

    total = len(test_cases)
    results: list[CaseResult] = []
    suite_start = time.time()

    for i, tc in enumerate(test_cases, 1):
        if verbose:
            desc = tc.get("description", tc["case_id"])
            print(f"\n{'='*60}")
            print(f"[{i}/{total}] {tc['case_id']}: {desc}")
            print(f"{'='*60}")

        result = await evaluator(tc)
        results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"  {'✅' if result.passed else '❌'} {status}  ({result.elapsed_s:.1f}s)")
            if result.error:
                print(f"  ERROR: {result.error}")
            for k, v in result.metrics.items():
                print(f"    {k}: {v}")

    passed = sum(1 for r in results if r.passed)
    suite_elapsed = time.time() - suite_start

    # Aggregate metrics — average all numeric metrics
    aggregate: dict = {
        "pass_rate": round(passed / total, 2) if total else 0,
        "total_time_s": round(suite_elapsed, 1),
    }
    numeric_keys: set[str] = set()
    for r in results:
        for k, v in r.metrics.items():
            if isinstance(v, (int, float)):
                numeric_keys.add(k)
    for k in sorted(numeric_keys):
        vals = [r.metrics[k] for r in results if k in r.metrics]
        aggregate[f"avg_{k}"] = round(sum(vals) / len(vals), 2) if vals else 0

    return SuiteResult(
        suite_name=suite_name,
        total=total,
        passed=passed,
        failed=total - passed,
        elapsed_s=round(suite_elapsed, 1),
        case_results=results,
        aggregate_metrics=aggregate,
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def print_summary(suite_result: SuiteResult) -> None:
    sr = suite_result
    print(f"\n{'='*60}")
    print(f"SUITE: {sr.suite_name} — {sr.passed}/{sr.total} PASS ({sr.elapsed_s:.1f}s)")
    print(f"{'='*60}")
    for k, v in sr.aggregate_metrics.items():
        print(f"  {k}: {v}")


def save_json_report(suite_results: list[SuiteResult], output_path: Path) -> None:
    data = []
    for sr in suite_results:
        entry = {
            "suite": sr.suite_name,
            "total": sr.total,
            "passed": sr.passed,
            "failed": sr.failed,
            "elapsed_s": sr.elapsed_s,
            "aggregate_metrics": sr.aggregate_metrics,
            "cases": [asdict(cr) for cr in sr.case_results],
        }
        data.append(entry)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="NS-TEA Evaluation Harness v2")
    parser.add_argument("--suite", choices=list(SUITES.keys()), help="Suite to run (default: all)")
    parser.add_argument("--report", choices=["text", "json", "both"], default="both")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case output")
    args = parser.parse_args()

    suites_to_run = [args.suite] if args.suite else list(SUITES.keys())
    all_results: list[SuiteResult] = []

    for suite_name in suites_to_run:
        print(f"\n{'#'*60}")
        print(f"# Running: {suite_name} — {SUITES[suite_name]['description']}")
        print(f"{'#'*60}")
        result = await run_suite(suite_name, verbose=not args.quiet)
        all_results.append(result)

    if args.report in ("text", "both"):
        for sr in all_results:
            print_summary(sr)

    if args.report in ("json", "both"):
        output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "evaluation_results.json"
        save_json_report(all_results, output_path)


if __name__ == "__main__":
    asyncio.run(main())
