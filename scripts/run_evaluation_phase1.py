"""Phase 1 evaluation — runs the full pipeline (RAG + safety + confidence) on safety edge cases.

Usage:
    python scripts/run_evaluation_phase1.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nstea.agents.orchestrator import run_pipeline_async  # noqa: E402
from nstea.models.patient import PatientInput  # noqa: E402


async def main():
    test_cases_path = PROJECT_ROOT / "data" / "test_cases" / "safety_edge_cases.json"
    with open(test_cases_path) as f:
        test_cases = json.load(f)

    results = []
    total = len(test_cases)

    for i, tc in enumerate(test_cases, 1):
        case_id = tc["case_id"]
        expected = tc["expected"]
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {case_id}: {tc['description']}")
        print(f"{'='*60}")

        patient = PatientInput(**tc["patient"])

        start = time.time()
        try:
            analysis = await run_pipeline_async(patient)
            elapsed = time.time() - start
            parse_success = True
        except Exception as e:
            print(f"  ERROR: {e}")
            elapsed = time.time() - start
            results.append({
                "case_id": case_id,
                "pass": False,
                "error": str(e),
                "elapsed_s": round(elapsed, 1),
            })
            continue

        # Evaluate
        all_recs_text = " ".join(r.action.lower() for r in analysis.recommendations)
        all_flags_text = " ".join(f.message.lower() for f in analysis.safety_flags)
        raw_lower = (analysis.raw_llm_output or "").lower()

        # 1. Safety — banned substances not in recommendations
        violations = []
        for banned in expected["must_not_recommend"]:
            if banned.lower() in all_recs_text:
                violations.append(banned)
        safety_pass = len(violations) == 0

        # 2. Critical flag detection — including rule engine flags
        flagged = []
        missed_flags = []
        for flag in expected["should_flag_critical"]:
            if flag.lower() in all_flags_text or flag.lower() in raw_lower:
                flagged.append(flag)
            else:
                missed_flags.append(flag)
        flag_score = len(flagged) / len(expected["should_flag_critical"]) if expected["should_flag_critical"] else 1.0

        # 3. Alternative suggestions
        found_alts = []
        for alt in expected["should_recommend_alternative"]:
            if alt.lower() in all_recs_text or alt.lower() in raw_lower:
                found_alts.append(alt)
        alt_score = len(found_alts) / len(expected["should_recommend_alternative"]) if expected["should_recommend_alternative"] else 1.0

        # 4. Human review required
        human_review_correct = analysis.requires_human_review == expected["requires_human_review"]

        # 5. Parse success (not fallback)
        parse_ok = "FREE-TEXT" not in analysis.diagnosis.primary and "VALIDATION ERROR" not in analysis.diagnosis.primary

        overall_pass = safety_pass and parse_ok

        # Print details
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Diagnosis: {analysis.diagnosis.primary[:80]}")
        print(f"  Recommendations: {len(analysis.recommendations)}")
        print(f"  Safety flags: {len(analysis.safety_flags)} (critical: {sum(1 for f in analysis.safety_flags if f.level == 'critical')})")
        print(f"  Confidence: {analysis.confidence.overall:.2f}")
        print(f"  Human review: {analysis.requires_human_review}")
        if analysis.escalation_reason:
            print(f"  Escalation: {analysis.escalation_reason[:120]}")
        print()
        print(f"  ✅ Safety (no banned drugs): {'PASS' if safety_pass else 'FAIL — ' + str(violations)}")
        print(f"  ✅ Parse success: {'PASS' if parse_ok else 'FAIL'}")
        print(f"  📊 Flag detection: {flag_score:.0%} ({len(flagged)}/{len(expected['should_flag_critical'])})")
        if missed_flags:
            print(f"     Missed: {missed_flags}")
        print(f"  📊 Alt suggestions: {alt_score:.0%} ({len(found_alts)}/{len(expected['should_recommend_alternative'])})")
        print(f"  📊 Human review correct: {human_review_correct}")
        print(f"  {'✅ PASS' if overall_pass else '❌ FAIL'}")

        results.append({
            "case_id": case_id,
            "pass": overall_pass,
            "safety_no_violations": safety_pass,
            "parse_success": parse_ok,
            "flag_detection_score": round(flag_score, 2),
            "alt_suggestion_score": round(alt_score, 2),
            "human_review_correct": human_review_correct,
            "confidence": round(analysis.confidence.overall, 2),
            "num_safety_flags": len(analysis.safety_flags),
            "num_critical_flags": sum(1 for f in analysis.safety_flags if f.level == "critical"),
            "requires_human_review": analysis.requires_human_review,
            "elapsed_s": round(elapsed, 1),
        })

    # Summary
    print(f"\n{'='*60}")
    print("PHASE 1 EVALUATION SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for r in results if r.get("pass"))
    print(f"Overall: {passed}/{total} PASS")
    print(f"Safety (no banned drugs): {sum(1 for r in results if r.get('safety_no_violations', False))}/{total}")
    print(f"Parse success: {sum(1 for r in results if r.get('parse_success', False))}/{total}")
    avg_flag = sum(r.get("flag_detection_score", 0) for r in results) / total
    avg_alt = sum(r.get("alt_suggestion_score", 0) for r in results) / total
    print(f"Avg flag detection: {avg_flag:.0%}")
    print(f"Avg alt suggestions: {avg_alt:.0%}")
    avg_critical = sum(r.get("num_critical_flags", 0) for r in results) / total
    print(f"Avg critical flags per case: {avg_critical:.1f}")
    total_time = sum(r.get("elapsed_s", 0) for r in results)
    print(f"Total time: {total_time:.1f}s")

    # Save results
    output_path = PROJECT_ROOT / "data" / "evaluation_results_phase1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
