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


def _semantic_pair_evaluable(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> bool:
    """Return whether a repair transition has a measurable semantic outcome.

    A failed failure-stage generation is not a valid pre-repair outcome.
    Semantic judge errors are measurement failures at either stage. A failed
    repair-stage generation remains a real unsuccessful repair and is therefore
    retained as semantic incorrect.
    """
    return bool(
        not before.get("generation_failed", False)
        and not before.get("semantic_judge_error", False)
        and not after.get("semantic_judge_error", False)
    )


def _supporting_ids(trace: Dict[str, Any]) -> set:
    return {
        str(item.get("passage_id", ""))
        for item in trace.get("selected_evidence", [])
        if item.get("is_supporting", False)
        and item.get("passage_id")
    }


def _failure_history_ids(
    trace: Dict[str, Any],
    failure: str,
    field: str,
) -> set:
    values = set()
    for item in trace.get("failure_history", []):
        if item.get("failure") != failure:
            continue
        value = item.get(field, [])
        if isinstance(value, list):
            values.update(str(entry) for entry in value if entry)
        elif value:
            values.add(str(value))
    return values


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

    paired_before_sem = []
    paired_after_sem = []
    semantic_gain_values = []
    recovered = 0
    regressed = 0
    stayed_correct = 0
    stayed_wrong = 0
    semantic_indeterminate = 0

    remaining = Counter()
    new_failures = Counter()
    new_failure_detected = 0
    regression_detected = 0
    new_failure_values = []
    regression_detected_values = []

    before_generation_failures = 0
    after_generation_failures = 0
    before_judge_errors = 0
    after_judge_errors = 0
    outcome_indeterminate_count = 0

    m_context_restored = 0
    m_context_size_restored = 0
    m_added_passages = 0
    d_removed = 0
    d_correct_evidence_removed = 0
    d_unresolved = 0
    d_detector_failed = 0
    g_answer_unchanged = 0
    g_answer_regenerated = 0
    g_generation_failed = 0
    g_dependency_restored = 0
    g_restoration_mismatch = 0

    for trace in traces:
        before = _before(trace)
        after = _after(trace)

        before_em.append(float(bool(before.get("exact_match", False))))
        after_em.append(float(bool(after.get("exact_match", False))))
        before_f1.append(float(before.get("token_f1", 0.0)))
        after_f1.append(float(after.get("token_f1", 0.0)))

        before_generation_failures += bool(
            before.get("generation_failed", False)
        )
        after_generation_failures += bool(
            after.get("generation_failed", False)
        )
        before_judge_errors += bool(
            before.get("semantic_judge_error", False)
        )
        after_judge_errors += bool(
            after.get("semantic_judge_error", False)
        )

        if _semantic_pair_evaluable(before, after):
            before_correct = bool(before.get("semantic_correct", False))
            after_correct = bool(after.get("semantic_correct", False))
            paired_before_sem.append(float(before_correct))
            paired_after_sem.append(float(after_correct))
            semantic_gain_values.append(
                float(after_correct) - float(before_correct)
            )

            if not before_correct and after_correct:
                recovered += 1
            elif before_correct and not after_correct:
                regressed += 1
            elif before_correct and after_correct:
                stayed_correct += 1
            else:
                stayed_wrong += 1
        else:
            semantic_indeterminate += 1

        remaining.update(trace.get("failure_after_repair", []))
        trace_new_failures = trace.get("new_failures_after_repair", [])
        new_failures.update(trace_new_failures)
        new_detected = bool(trace_new_failures)
        detected = bool(trace.get("regression_detected", False))
        new_failure_detected += new_detected
        regression_detected += detected
        new_failure_values.append(float(new_detected))
        regression_detected_values.append(float(detected))

        diagnosis = trace.get("repair_diagnosis", {})
        outcome_indeterminate_count += bool(
            diagnosis.get("outcome_indeterminate", False)
        )

        final_ids = {
            str(document.get("passage_id", ""))
            for document in trace.get("retrieval_events", [])
            if document.get("passage_id")
        }
        supporting_ids = _supporting_ids(trace)
        injected_missing_ids = _failure_history_ids(
            trace,
            "M",
            "removed_passage_ids",
        )
        injected_distractor_ids = _failure_history_ids(
            trace,
            "D",
            "distractor_passage_id",
        )

        if injected_missing_ids:
            m_context_restored += injected_missing_ids.issubset(final_ids)
        if injected_distractor_ids:
            d_removed += injected_distractor_ids.isdisjoint(final_ids)

        for item in trace.get("repair_history", []):
            if item.get("failure") == "M":
                m_context_size_restored += bool(
                    item.get("context_restored", False)
                )
                m_added_passages += len(
                    item.get("recovered_passage_ids", [])
                )
            elif item.get("failure") == "D":
                removed_ids = {
                    str(passage_id)
                    for passage_id in item.get("removed_passage_ids", [])
                    if passage_id
                }
                d_correct_evidence_removed += bool(
                    (removed_ids & supporting_ids) - final_ids
                )
                d_unresolved += bool(
                    item.get("unresolved_conflict", False)
                )
                d_detector_failed += bool(
                    item.get("detector_failed", False)
                )
            elif item.get("failure") == "G":
                answer_unchanged = bool(item.get("answer_kept", False))
                answer_regenerated = item.get("answer_regenerated")
                if answer_regenerated is None:
                    answer_regenerated = bool(
                        item.get("generation_pending") is False
                        and not item.get("generation_failed", False)
                    )

                g_answer_unchanged += answer_unchanged
                g_answer_regenerated += bool(answer_regenerated)
                g_generation_failed += bool(
                    item.get("generation_failed", False)
                )

                restored_step_id = str(item.get("restored_step_id", ""))
                restored_dependency = str(
                    item.get("restored_dependency", "")
                )
                expected_step_id = str(
                    trace.get("g_corrupted_step_id", "")
                )
                expected_dependency = str(
                    trace.get("g_removed_dependency", "")
                )

                restoration_matches = bool(
                    item.get("repair") == "restore_reasoning_dependency"
                    and restored_step_id == expected_step_id
                    and restored_dependency == expected_dependency
                    and restored_step_id
                    and restored_dependency
                    and item.get("dependency_restored", False)
                )
                g_dependency_restored += restoration_matches
                g_restoration_mismatch += not restoration_matches

    semantic_evaluable_count = len(paired_before_sem)
    before_wrong = semantic_evaluable_count - int(sum(paired_before_sem))
    before_correct_count = int(sum(paired_before_sem))

    recovery_values = [
        float(not bool(before_value) and bool(after_value))
        for before_value, after_value in zip(
            paired_before_sem,
            paired_after_sem,
        )
        if not bool(before_value)
    ]
    regression_values = [
        float(bool(before_value) and not bool(after_value))
        for before_value, after_value in zip(
            paired_before_sem,
            paired_after_sem,
        )
        if bool(before_value)
    ]

    trace_ids = [str(trace.get("trace_id", "")) for trace in traces]
    nonempty_trace_ids = [trace_id for trace_id in trace_ids if trace_id]
    duplicate_trace_count = len(nonempty_trace_ids) - len(
        set(nonempty_trace_ids)
    )
    missing_trace_id_count = total - len(nonempty_trace_ids)

    fully_repaired_count = sum(
        not trace.get("failure_after_repair", [])
        and not trace.get("repair_diagnosis", {}).get(
            "outcome_indeterminate",
            False,
        )
        for trace in traces
    )

    return {
        "total": total,
        "duplicate_trace_count": duplicate_trace_count,
        "missing_trace_id_count": missing_trace_id_count,
        "before_em_accuracy": sum(before_em) / total if total else 0.0,
        "after_em_accuracy": sum(after_em) / total if total else 0.0,
        "em_gain": (sum(after_em) - sum(before_em)) / total if total else 0.0,
        "before_mean_token_f1": sum(before_f1) / total if total else 0.0,
        "after_mean_token_f1": sum(after_f1) / total if total else 0.0,
        "token_f1_gain": (sum(after_f1) - sum(before_f1)) / total if total else 0.0,
        "semantic_evaluable_count": semantic_evaluable_count,
        "semantic_indeterminate_count": semantic_indeterminate,
        "before_semantic_accuracy": (
            sum(paired_before_sem) / semantic_evaluable_count
            if semantic_evaluable_count
            else 0.0
        ),
        "after_semantic_accuracy": (
            sum(paired_after_sem) / semantic_evaluable_count
            if semantic_evaluable_count
            else 0.0
        ),
        "semantic_gain": (
            (sum(paired_after_sem) - sum(paired_before_sem))
            / semantic_evaluable_count
            if semantic_evaluable_count
            else 0.0
        ),
        "semantic_gain_ci95": bootstrap_mean_ci(semantic_gain_values),
        "semantic_recovered_count": recovered,
        "semantic_regressed_count": regressed,
        "semantic_stayed_correct": stayed_correct,
        "semantic_stayed_wrong": stayed_wrong,
        "semantic_recovery_denominator": before_wrong,
        "semantic_recovery_rate": recovered / before_wrong if before_wrong else 0.0,
        "semantic_recovery_rate_ci95": bootstrap_mean_ci(recovery_values),
        "semantic_regression_denominator": before_correct_count,
        "semantic_regression_rate": (
            regressed / before_correct_count if before_correct_count else 0.0
        ),
        "semantic_regression_rate_ci95": bootstrap_mean_ci(regression_values),
        "remaining_M_count": remaining["M"],
        "remaining_D_count": remaining["D"],
        "remaining_G_count": remaining["G"],
        "fully_repaired_count": fully_repaired_count,
        "new_M_count": new_failures["M"],
        "new_D_count": new_failures["D"],
        "new_G_count": new_failures["G"],
        "new_failure_count": new_failure_detected,
        "new_failure_rate": (
            new_failure_detected / total if total else 0.0
        ),
        "new_failure_rate_ci95": bootstrap_mean_ci(new_failure_values),
        "regression_detected_count": regression_detected,
        "regression_detected_rate": (
            regression_detected / total if total else 0.0
        ),
        "regression_detected_rate_ci95": bootstrap_mean_ci(
            regression_detected_values
        ),
        "before_generation_failure_count": before_generation_failures,
        "generation_failure_count": after_generation_failures,
        "before_semantic_judge_error_count": before_judge_errors,
        "semantic_judge_error_count": after_judge_errors,
        "outcome_indeterminate_count": outcome_indeterminate_count,
        "M_context_restored_count": m_context_restored,
        "M_injected_passage_restored_count": m_context_restored,
        "M_context_size_restored_count": m_context_size_restored,
        "M_added_passage_count": m_added_passages,
        "D_distractor_removed_count": d_removed,
        "D_correct_evidence_removed_count": d_correct_evidence_removed,
        "D_unresolved_conflict_count": d_unresolved,
        "D_detector_failed_count": d_detector_failed,
        "G_answer_kept_count": g_answer_unchanged,
        "G_answer_unchanged_count": g_answer_unchanged,
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
    condition_names = (first, second, name)
    maps = {
        condition: _trace_map(results[condition])
        for condition in condition_names
    }
    common_ids = set.intersection(
        *(set(mapping) for mapping in maps.values())
    )
    common_ids.discard("")
    if not common_ids:
        return {"common_trace_count": 0}

    comparison = {}
    for condition in condition_names:
        subset = [maps[condition][trace_id] for trace_id in sorted(common_ids)]
        metrics = calculate_metrics(subset)
        comparison[condition] = {
            "after_semantic_accuracy": metrics["after_semantic_accuracy"],
            "semantic_gain": metrics["semantic_gain"],
            "semantic_recovery_rate": metrics["semantic_recovery_rate"],
            "semantic_regression_rate": metrics["semantic_regression_rate"],
            "new_failure_rate": metrics["new_failure_rate"],
            "semantic_evaluable_count": metrics["semantic_evaluable_count"],
        }

    joint_failure_ids = []
    joint_correct_ids = []
    for trace_id in sorted(common_ids):
        pairs = [
            (_before(maps[condition][trace_id]), _after(maps[condition][trace_id]))
            for condition in condition_names
        ]
        if not all(
            _semantic_pair_evaluable(before, after)
            for before, after in pairs
        ):
            continue

        before_values = [
            bool(before.get("semantic_correct", False))
            for before, _ in pairs
        ]
        if all(not value for value in before_values):
            joint_failure_ids.append(trace_id)
        if all(before_values):
            joint_correct_ids.append(trace_id)

    joint_recovery = {}
    for condition in condition_names:
        if joint_failure_ids:
            joint_recovery[condition] = sum(
                bool(
                    _after(maps[condition][trace_id]).get(
                        "semantic_correct",
                        False,
                    )
                )
                for trace_id in joint_failure_ids
            ) / len(joint_failure_ids)
        else:
            joint_recovery[condition] = 0.0

    joint_regression = {}
    for condition in condition_names:
        if joint_correct_ids:
            joint_regression[condition] = sum(
                not bool(
                    _after(maps[condition][trace_id]).get(
                        "semantic_correct",
                        False,
                    )
                )
                for trace_id in joint_correct_ids
            ) / len(joint_correct_ids)
        else:
            joint_regression[condition] = 0.0

    return {
        "common_trace_count": len(common_ids),
        "conditions": comparison,
        "extra_semantic_gain_vs_best_single": comparison[name]["semantic_gain"]
        - max(
            comparison[first]["semantic_gain"],
            comparison[second]["semantic_gain"],
        ),
        "joint_failure_trace_count": len(joint_failure_ids),
        "joint_failure_recovery_rates": joint_recovery,
        "extra_recovery_vs_best_single": joint_recovery[name]
        - max(joint_recovery[first], joint_recovery[second]),
        "joint_correct_trace_count": len(joint_correct_ids),
        "joint_correct_regression_rates": joint_regression,
        "extra_regression_vs_worst_single": joint_regression[name]
        - max(joint_regression[first], joint_regression[second]),
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