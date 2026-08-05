import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42

EXPERIMENTS = {
    "M": "data/repaired_result/M_repair_results.json",
    "D": "data/repaired_result/D_repair_results.json",
    "G": "data/repaired_result/G_repair_results.json",
    "M_D": "data/repaired_result/M_D_repair_results.json",
    "M_G": "data/repaired_result/M_G_repair_results.json",
    "D_G": "data/repaired_result/D_G_repair_results.json",
}

COMPOUNDS = {
    "M_D": ("M", "D"),
    "M_G": ("M", "G"),
    "D_G": ("D", "G"),
}


def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _before(trace: Dict[str, Any]) -> Dict[str, Any]:
    return trace.get("failure_evaluation") or {}


def _after(trace: Dict[str, Any]) -> Dict[str, Any]:
    return trace.get("repair_evaluation") or trace.get("evaluation", {})


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def bootstrap_mean_ci(values: List[float]) -> List[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(sample) / len(sample))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def calculate_metrics(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(traces)
    before_em = []
    after_em = []
    before_f1 = []
    after_f1 = []
    before_sem = []
    after_sem = []

    recovered = 0
    regressed = 0
    stayed_correct = 0
    stayed_wrong = 0

    remaining = Counter()
    new_failures = Counter()
    regression_detected = 0
    generation_failures = 0
    judge_errors = 0

    m_context_restored = 0
    m_added_passages = 0
    d_removed = 0
    d_correct_evidence_removed = 0
    d_unresolved = 0
    d_detector_failed = 0
    g_answer_kept = 0
    g_answer_regenerated = 0
    g_generation_failed = 0
    g_dependency_restored = 0
    g_restoration_mismatch = 0

    semantic_gain_values = []
    new_failure_values = []

    for trace in traces:
        before = _before(trace)
        after = _after(trace)
        before_exact = bool(before.get("exact_match", False))
        after_exact = bool(after.get("exact_match", False))
        before_correct = bool(before.get("semantic_correct", False))
        after_correct = bool(after.get("semantic_correct", False))

        before_em.append(float(before_exact))
        after_em.append(float(after_exact))
        before_f1.append(float(before.get("token_f1", 0.0)))
        after_f1.append(float(after.get("token_f1", 0.0)))
        before_sem.append(float(before_correct))
        after_sem.append(float(after_correct))
        semantic_gain_values.append(float(after_correct) - float(before_correct))

        if not before_correct and after_correct:
            recovered += 1
        elif before_correct and not after_correct:
            regressed += 1
        elif before_correct and after_correct:
            stayed_correct += 1
        else:
            stayed_wrong += 1

        remaining.update(trace.get("failure_after_repair", []))
        new_failures.update(trace.get("new_failures_after_repair", []))
        detected = bool(trace.get("regression_detected", False))
        regression_detected += detected
        new_failure_values.append(float(detected))
        generation_failures += bool(after.get("generation_failed", False))
        judge_errors += bool(after.get("semantic_judge_error", False))

        final_ids = {
            str(document.get("passage_id", ""))
            for document in trace.get("retrieval_events", [])
        }
        failure_d = next(
            (
                item
                for item in trace.get("failure_history", [])
                if item.get("failure") == "D"
            ),
            None,
        )

        for item in trace.get("repair_history", []):
            if item.get("failure") == "M":
                m_context_restored += bool(item.get("context_restored", False))
                m_added_passages += len(item.get("recovered_passage_ids", []))
            elif item.get("failure") == "D":
                removed_ids = set(item.get("removed_passage_ids", []))
                d_unresolved += bool(item.get("unresolved_conflict", False))
                d_detector_failed += bool(item.get("detector_failed", False))
                if failure_d:
                    distractor_id = failure_d.get("distractor_passage_id")
                    source_id = failure_d.get("source_passage_id")
                    d_removed += bool(distractor_id and distractor_id not in final_ids)
                    d_correct_evidence_removed += bool(
                        source_id and source_id not in final_ids
                    )
                elif removed_ids:
                    d_removed += 1
            elif item.get("failure") == "G":
                g_answer_kept += bool(item.get("answer_kept", False))
                g_answer_regenerated += bool(item.get("repair_applied", False))
                g_generation_failed += bool(item.get("generation_failed", False))

                restored_step_id = str(item.get("restored_step_id", ""))
                restored_dependency = str(item.get("restored_dependency", ""))
                expected_step_id = str(trace.get("g_corrupted_step_id", ""))
                expected_dependency = str(trace.get("g_removed_dependency", ""))

                restoration_matches = (
                    item.get("repair") == "restore_reasoning_dependency"
                    and restored_step_id == expected_step_id
                    and restored_dependency == expected_dependency
                    and bool(restored_step_id)
                    and bool(restored_dependency)
                )
                g_dependency_restored += restoration_matches
                g_restoration_mismatch += not restoration_matches

    before_wrong = total - int(sum(before_sem))
    before_correct_count = int(sum(before_sem))
    recovery_values = [
        1.0 if (not bool(_before(t).get("semantic_correct", False)) and bool(_after(t).get("semantic_correct", False))) else 0.0
        for t in traces
        if not bool(_before(t).get("semantic_correct", False))
    ]
    regression_values = [
        1.0 if (bool(_before(t).get("semantic_correct", False)) and not bool(_after(t).get("semantic_correct", False))) else 0.0
        for t in traces
        if bool(_before(t).get("semantic_correct", False))
    ]

    duplicate_trace_count = total - len({str(trace.get("trace_id", "")) for trace in traces})

    return {
        "total": total,
        "duplicate_trace_count": duplicate_trace_count,
        "before_em_accuracy": sum(before_em) / total if total else 0.0,
        "after_em_accuracy": sum(after_em) / total if total else 0.0,
        "em_gain": (sum(after_em) - sum(before_em)) / total if total else 0.0,
        "before_mean_token_f1": sum(before_f1) / total if total else 0.0,
        "after_mean_token_f1": sum(after_f1) / total if total else 0.0,
        "token_f1_gain": (sum(after_f1) - sum(before_f1)) / total if total else 0.0,
        "before_semantic_accuracy": sum(before_sem) / total if total else 0.0,
        "after_semantic_accuracy": sum(after_sem) / total if total else 0.0,
        "semantic_gain": (sum(after_sem) - sum(before_sem)) / total if total else 0.0,
        "semantic_gain_ci95": bootstrap_mean_ci(semantic_gain_values),
        "semantic_recovered_count": recovered,
        "semantic_regressed_count": regressed,
        "semantic_stayed_correct": stayed_correct,
        "semantic_stayed_wrong": stayed_wrong,
        "semantic_recovery_rate": recovered / before_wrong if before_wrong else 0.0,
        "semantic_recovery_rate_ci95": bootstrap_mean_ci(recovery_values),
        "semantic_regression_rate": (
            regressed / before_correct_count if before_correct_count else 0.0
        ),
        "semantic_regression_rate_ci95": bootstrap_mean_ci(regression_values),
        "remaining_M_count": remaining["M"],
        "remaining_D_count": remaining["D"],
        "remaining_G_count": remaining["G"],
        "fully_repaired_count": sum(
            not trace.get("failure_after_repair", []) for trace in traces
        ),
        "new_M_count": new_failures["M"],
        "new_D_count": new_failures["D"],
        "new_G_count": new_failures["G"],
        "regression_detected_count": regression_detected,
        "new_failure_rate": regression_detected / total if total else 0.0,
        "new_failure_rate_ci95": bootstrap_mean_ci(new_failure_values),
        "generation_failure_count": generation_failures,
        "semantic_judge_error_count": judge_errors,
        "M_context_restored_count": m_context_restored,
        "M_added_passage_count": m_added_passages,
        "D_distractor_removed_count": d_removed,
        "D_correct_evidence_removed_count": d_correct_evidence_removed,
        "D_unresolved_conflict_count": d_unresolved,
        "D_detector_failed_count": d_detector_failed,
        "G_answer_kept_count": g_answer_kept,
        "G_answer_regenerated_count": g_answer_regenerated,
        "G_generation_failed_count": g_generation_failed,
        "G_dependency_restored_count": g_dependency_restored,
        "G_restoration_mismatch_count": g_restoration_mismatch,
    }


def _trace_map(traces: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(trace.get("trace_id", "")): trace for trace in traces}


def compound_comparison(
    name: str,
    results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    first, second = COMPOUNDS[name]
    common_ids = (
        set(_trace_map(results[name]))
        & set(_trace_map(results[first]))
        & set(_trace_map(results[second]))
    )
    if not common_ids:
        return {"common_trace_count": 0}

    comparison = {}
    for condition in (first, second, name):
        subset = [
            trace
            for trace in results[condition]
            if str(trace.get("trace_id", "")) in common_ids
        ]
        metrics = calculate_metrics(subset)
        comparison[condition] = {
            "semantic_recovery_rate": metrics["semantic_recovery_rate"],
            "semantic_regression_rate": metrics["semantic_regression_rate"],
            "new_failure_rate": metrics["new_failure_rate"],
        }

    return {
        "common_trace_count": len(common_ids),
        "conditions": comparison,
        "extra_recovery_vs_best_single": comparison[name]["semantic_recovery_rate"]
        - max(
            comparison[first]["semantic_recovery_rate"],
            comparison[second]["semantic_recovery_rate"],
        ),
        "extra_regression_vs_worst_single": comparison[name]["semantic_regression_rate"]
        - max(
            comparison[first]["semantic_regression_rate"],
            comparison[second]["semantic_regression_rate"],
        ),
    }


def save_csv(summary: Dict[str, Any], output_path: Path) -> None:
    rows = []
    for experiment, metrics in summary.items():
        row = {"experiment": experiment}
        for key, value in metrics.items():
            if not isinstance(value, (dict, list)):
                row[key] = value
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    results = {}
    for experiment, path in EXPERIMENTS.items():
        if Path(path).exists():
            results[experiment] = load_results(path)

    if not results:
        raise FileNotFoundError("No repair result files were found.")

    summary = {
        experiment: calculate_metrics(traces)
        for experiment, traces in results.items()
    }
    for name in COMPOUNDS:
        if name in results and all(item in results for item in COMPOUNDS[name]):
            summary[name]["compound_comparison"] = compound_comparison(name, results)

    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "repair_summary.json"
    csv_path = output_dir / "repair_summary.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    save_csv(summary, csv_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()