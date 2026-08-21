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
    "L": "data/repaired_result/L_repair_results.json",
    "M_D_fixed": "data/repaired_result/M_D_repair_results.json",
    "M_D_reverse": "data/repaired_result/M_D_reverse_repair_results.json",
    "M_D_safe": "data/repaired_result/M_D_safe_repair_results.json",
    "M_L_fixed": "data/repaired_result/M_L_repair_results.json",
    "M_L_reverse": "data/repaired_result/M_L_reverse_repair_results.json",
    "M_L_safe": "data/repaired_result/M_L_safe_repair_results.json",
    "D_L_fixed": "data/repaired_result/D_L_repair_results.json",
    "D_L_reverse": "data/repaired_result/D_L_reverse_repair_results.json",
    "D_L_safe": "data/repaired_result/D_L_safe_repair_results.json",
    "M_D_L_fixed": "data/repaired_result/M_D_L_repair_results.json",
    "M_D_L_reverse": "data/repaired_result/M_D_L_reverse_repair_results.json",
    "M_D_L_safe": "data/repaired_result/M_D_L_safe_repair_results.json",
    "M_D_b2": "data/repaired_result/M_D_b2_repair_results.json",
    "M_D_b3": "data/repaired_result/M_D_b3_repair_results.json",
    "M_L_b2": "data/repaired_result/M_L_b2_repair_results.json",
    "M_L_b3": "data/repaired_result/M_L_b3_repair_results.json",
    "D_L_b2": "data/repaired_result/D_L_b2_repair_results.json",
    "D_L_b3": "data/repaired_result/D_L_b3_repair_results.json",
    "M_D_L_b2": "data/repaired_result/M_D_L_b2_repair_results.json",
    "M_D_L_b3": "data/repaired_result/M_D_L_b3_repair_results.json",
}

FAILURE_RESULTS = {
    "M_D": "data/results/compound/M_D_results.json",
    "M_L": "data/results/compound/M_L_results.json",
    "D_L": "data/results/compound/D_L_results.json",
    "M_D_L": "data/results/compound/M_D_L_results.json",
}

OPTIONAL_BASELINE_METHODS = {
    "M_D": {"b2": "M_D_b2", "b3": "M_D_b3"},
    "M_L": {"b2": "M_L_b2", "b3": "M_L_b3"},
    "D_L": {"b2": "D_L_b2", "b3": "D_L_b3"},
    "M_D_L": {"b2": "M_D_L_b2", "b3": "M_D_L_b3"},
}

METHOD_GROUPS = {
    "M_D": {
        "fixed": "M_D_fixed",
        "reverse": "M_D_reverse",
        "safe": "M_D_safe",
    },
    "M_L": {
        "fixed": "M_L_fixed",
        "reverse": "M_L_reverse",
        "safe": "M_L_safe",
    },
    "D_L": {
        "fixed": "D_L_fixed",
        "reverse": "D_L_reverse",
        "safe": "D_L_safe",
    },
    "M_D_L": {
        "fixed": "M_D_L_fixed",
        "reverse": "M_D_L_reverse",
        "safe": "M_D_L_safe",
    },
}


def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _before(trace: Dict[str, Any]) -> Dict[str, Any]:
    return trace.get("failure_evaluation") or trace.get("evaluation", {})


def _after(trace: Dict[str, Any]) -> Dict[str, Any]:
    return trace.get("repair_evaluation") or trace.get("evaluation", {})


def _evaluation_evaluable(evaluation: Dict[str, Any]) -> bool:
    return bool(
        not evaluation.get("generation_failed", False)
        and not evaluation.get("semantic_judge_error", False)
    )


def _trace_timed_out(trace: Dict[str, Any]) -> bool:
    """
    Detect a timeout only when a saved trace explicitly records one.

    If your runner completely skips timed-out traces, they will simply be
    absent from the result file. method_comparison() handles those by taking
    the trace-ID intersection across fixed/reverse/Safe.

    This helper intentionally avoids treating a configured timeout duration
    (for example timeout=60) as an actual timeout event.
    """
    containers = [
        trace,
        trace.get("runtime", {}),
        trace.get("execution", {}),
        trace.get("technical", {}),
        trace.get("repair_evaluation", {}),
    ]

    true_flags = {
        "technical_timeout",
        "timed_out",
        "timeout_occurred",
        "repair_timeout",
    }

    for container in containers:
        if not isinstance(container, dict):
            continue

        for key in true_flags:
            if container.get(key) is True:
                return True

        for key in ("technical_status", "execution_status", "status"):
            value = str(container.get(key, "")).strip().lower()
            if value in {"timeout", "timed_out", "technical_timeout"}:
                return True

        error_value = str(container.get("technical_error", "")).strip().lower()
        if "timeout" in error_value:
            return True

    return False


def _pair_evaluable(
    trace: Dict[str, Any],
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> bool:
    return bool(
        not _trace_timed_out(trace)
        and _evaluation_evaluable(before)
        and _evaluation_evaluable(after)
    )


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round(fraction * (len(ordered) - 1)))),
    )
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


def _supporting_ids(trace: Dict[str, Any]) -> set:
    explicit = {
        str(item)
        for item in trace.get("gold_supporting_passage_ids", [])
        if item
    }
    if explicit:
        return explicit
    return {
        str(item.get("passage_id", ""))
        for item in trace.get("selected_evidence", [])
        if item.get("is_supporting", False) and item.get("passage_id")
    }


def _failure_history_ids(
    trace: Dict[str, Any], failure: str, field: str
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


def calculate_metrics(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(traces)

    before_em: List[float] = []
    after_em: List[float] = []
    before_f1: List[float] = []
    after_f1: List[float] = []
    before_sem: List[float] = []
    after_sem: List[float] = []
    gain_values: List[float] = []
    recovery_values: List[float] = []
    regression_values: List[float] = []

    recovered = 0
    regressed = 0
    stayed_correct = 0
    stayed_wrong = 0
    indeterminate = 0

    remaining = Counter()
    new_failures = Counter()
    regression_detected = 0
    new_failure_trace_count = 0
    fully_repaired = 0

    m_restored = 0
    d_removed = 0
    l_rewrite_applied = 0
    l_no_repair_needed = 0
    l_technical_errors = 0
    safe_kept_steps = 0
    safe_reverted_steps = 0
    safe_planned_orders = Counter()
    safe_planner_errors = 0
    safe_step_errors = 0

    saved_timeout_count = 0

    for trace in traces:
        before = _before(trace)
        after = _after(trace)

        if _trace_timed_out(trace):
            saved_timeout_count += 1

        # Keep the original full-run EM/F1 summaries for compatibility.
        # Technical timeout handling for fair cross-method comparison happens
        # in method_comparison(), where all methods use the same jointly
        # evaluable trace subset.
        before_em.append(float(bool(before.get("exact_match", False))))
        after_em.append(float(bool(after.get("exact_match", False))))
        before_f1.append(float(before.get("token_f1", 0.0)))
        after_f1.append(float(after.get("token_f1", 0.0)))

        if _pair_evaluable(trace, before, after):
            b = bool(before.get("semantic_correct", False))
            a = bool(after.get("semantic_correct", False))

            before_sem.append(float(b))
            after_sem.append(float(a))
            gain_values.append(float(a) - float(b))

            if not b:
                recovery_values.append(float(a))
            if b:
                regression_values.append(float(not a))

            if not b and a:
                recovered += 1
            elif b and not a:
                regressed += 1
            elif b and a:
                stayed_correct += 1
            else:
                stayed_wrong += 1

            if a and not trace.get("failure_after_repair", []):
                fully_repaired += 1
        else:
            indeterminate += 1

        remaining.update(trace.get("failure_after_repair", []))

        trace_new_failures = trace.get("new_failures_after_repair", [])
        new_failures.update(trace_new_failures)
        new_failure_trace_count += bool(trace_new_failures)
        regression_detected += bool(trace.get("regression_detected", False))

        final_ids = {
            str(item.get("passage_id", ""))
            for item in trace.get("retrieval_events", [])
            if item.get("passage_id")
        }

        missing_ids = _failure_history_ids(trace, "M", "removed_passage_ids")
        distractor_ids = _failure_history_ids(trace, "D", "distractor_passage_id")

        if missing_ids:
            m_restored += missing_ids.issubset(final_ids)
        if distractor_ids:
            d_removed += distractor_ids.isdisjoint(final_ids)

        for item in trace.get("repair_history", []):
            if item.get("failure") == "L":
                l_rewrite_applied += bool(item.get("rewrite_applied", False))
                l_no_repair_needed += item.get("repair_needed") is False
                l_technical_errors += bool(item.get("technical_error"))

        composer = trace.get("safe_composer_history", {})
        if composer:
            planned_order = "->".join(
                str(item) for item in composer.get("planned_order", [])
            )
            if planned_order:
                safe_planned_orders[planned_order] += 1
            safe_planner_errors += bool(composer.get("planner_technical_error"))

        for step in composer.get("steps", []):
            if step.get("keep_change"):
                safe_kept_steps += 1
            else:
                safe_reverted_steps += 1
            safe_step_errors += bool(step.get("technical_error"))

    evaluable = len(before_sem)
    before_wrong = evaluable - int(sum(before_sem))
    before_correct = int(sum(before_sem))

    trace_ids = [str(trace.get("trace_id", "")) for trace in traces]
    nonempty = [trace_id for trace_id in trace_ids if trace_id]

    repair_modes = Counter(str(trace.get("repair_mode", "")) for trace in traces)
    common_mode = repair_modes.most_common(1)[0][0] if repair_modes else ""

    return {
        "total": total,
        "saved_timeout_count": saved_timeout_count,
        "missing_trace_id_count": total - len(nonempty),
        "duplicate_trace_count": len(nonempty) - len(set(nonempty)),
        "repair_mode": common_mode,
        "before_em_accuracy": sum(before_em) / total if total else 0.0,
        "after_em_accuracy": sum(after_em) / total if total else 0.0,
        "em_gain": (sum(after_em) - sum(before_em)) / total if total else 0.0,
        "before_mean_token_f1": sum(before_f1) / total if total else 0.0,
        "after_mean_token_f1": sum(after_f1) / total if total else 0.0,
        "token_f1_gain": (sum(after_f1) - sum(before_f1)) / total if total else 0.0,
        "semantic_evaluable_count": evaluable,
        "semantic_indeterminate_count": indeterminate,
        "before_semantic_accuracy": sum(before_sem) / evaluable if evaluable else 0.0,
        "after_semantic_accuracy": sum(after_sem) / evaluable if evaluable else 0.0,
        "semantic_gain": (
            (sum(after_sem) - sum(before_sem)) / evaluable
            if evaluable
            else 0.0
        ),
        "semantic_gain_ci95": bootstrap_mean_ci(gain_values),
        "semantic_recovered_count": recovered,
        "semantic_recovery_denominator": before_wrong,
        "semantic_recovery_rate": recovered / before_wrong if before_wrong else 0.0,
        "semantic_recovery_rate_ci95": bootstrap_mean_ci(recovery_values),
        "semantic_regressed_count": regressed,
        "semantic_regression_denominator": before_correct,
        "semantic_regression_rate": (
            regressed / before_correct if before_correct else 0.0
        ),
        "semantic_regression_rate_ci95": bootstrap_mean_ci(regression_values),
        "semantic_stayed_correct": stayed_correct,
        "semantic_stayed_wrong": stayed_wrong,
        "remaining_M_count": remaining["M"],
        "remaining_D_count": remaining["D"],
        "remaining_L_count": remaining["L"],
        "fully_repaired_count": fully_repaired,
        "fully_repaired_rate": fully_repaired / evaluable if evaluable else 0.0,
        "new_M_count": new_failures["M"],
        "new_D_count": new_failures["D"],
        "new_failure_count": new_failure_trace_count,
        "new_failure_rate": new_failure_trace_count / total if total else 0.0,
        "regression_detected_count": regression_detected,
        "regression_detected_rate": (
            regression_detected / total if total else 0.0
        ),
        "generation_failure_count": sum(
            bool(_after(trace).get("generation_failed", False))
            for trace in traces
        ),
        "semantic_judge_error_count": sum(
            bool(_after(trace).get("semantic_judge_error", False))
            for trace in traces
        ),
        "M_injected_passage_restored_count": m_restored,
        "D_distractor_removed_count": d_removed,
        "L_rewrite_applied_count": l_rewrite_applied,
        "L_no_repair_needed_count": l_no_repair_needed,
        "L_technical_error_count": l_technical_errors,
        "safe_kept_step_count": safe_kept_steps,
        "safe_reverted_step_count": safe_reverted_steps,
        "safe_planned_order_counts": dict(safe_planned_orders),
        "safe_planner_technical_error_count": safe_planner_errors,
        "safe_step_technical_error_count": safe_step_errors,
    }


def _trace_map(traces: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(trace.get("trace_id", "")): trace
        for trace in traces
        if trace.get("trace_id")
    }


def _no_repair_row(
    failure_map: Dict[str, Dict[str, Any]],
    paired_ids: set,
    attempted_count: int,
) -> Dict[str, Any]:
    selected = [failure_map[trace_id] for trace_id in sorted(paired_ids)]
    evaluations = [_before(trace) for trace in selected]

    # paired_ids has already been restricted to evaluable traces.
    count = len(evaluations)
    semantic_correct = sum(
        bool(item.get("semantic_correct", False)) for item in evaluations
    )

    return {
        "method": "no_repair",
        "attempted_count": attempted_count,
        "present_result_count": attempted_count,
        "missing_result_count": 0,
        "saved_timeout_count": 0,
        "trace_count": count,
        "semantic_evaluable_count": count,
        "after_em_accuracy": (
            sum(bool(item.get("exact_match", False)) for item in evaluations) / count
            if count
            else 0.0
        ),
        "after_mean_token_f1": (
            sum(float(item.get("token_f1", 0.0)) for item in evaluations) / count
            if count
            else 0.0
        ),
        "after_semantic_accuracy": semantic_correct / count if count else 0.0,
        "semantic_recovery_rate": 0.0,
        "semantic_regression_rate": 0.0,
        "fully_repaired_rate": 0.0,
        "regression_detected_rate": 0.0,
    }


def method_comparison(
    condition: str,
    group: Dict[str, str],
    repair_results: Dict[str, List[Dict[str, Any]]],
    failure_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a fair paired matrix.

    1. Start from the condition's failure-result trace IDs.
    2. Keep only trace IDs present in every method included in the matrix.
       The core methods are fixed, reverse, and Safe; B2/B3 are added when
       their result files exist. This automatically removes traces completely
       skipped because of timeout in any included method.
    3. From those common-present IDs, keep only traces that are semantically
       evaluable in ALL included methods.
    4. Calculate every method row on exactly that same paired subset.
    """
    maps = {
        method: _trace_map(repair_results[experiment])
        for method, experiment in group.items()
    }
    failure_map = _trace_map(failure_results)

    attempted_ids = set(failure_map)
    attempted_count = len(attempted_ids)

    execution = {}
    for method, mapping in maps.items():
        present_ids = attempted_ids & set(mapping)
        saved_timeout_count = sum(
            _trace_timed_out(mapping[trace_id])
            for trace_id in present_ids
        )
        execution[method] = {
            "attempted_count": attempted_count,
            "present_result_count": len(present_ids),
            "missing_result_count": attempted_count - len(present_ids),
            "saved_timeout_count": saved_timeout_count,
        }

    # First remove traces that are absent from any method result file.
    common_present_ids = set(attempted_ids)
    for mapping in maps.values():
        common_present_ids &= set(mapping)

    # Then require the failure result + every repair method to be evaluable.
    paired_ids = set()
    for trace_id in common_present_ids:
        failure_trace = failure_map[trace_id]
        failure_eval = _before(failure_trace)

        if not _evaluation_evaluable(failure_eval):
            continue

        all_methods_evaluable = True
        for mapping in maps.values():
            trace = mapping[trace_id]
            if not _pair_evaluable(trace, _before(trace), _after(trace)):
                all_methods_evaluable = False
                break

        if all_methods_evaluable:
            paired_ids.add(trace_id)

    rows = [
        _no_repair_row(
            failure_map=failure_map,
            paired_ids=paired_ids,
            attempted_count=attempted_count,
        )
    ]

    for method, experiment in group.items():
        subset = [maps[method][trace_id] for trace_id in sorted(paired_ids)]
        metrics = calculate_metrics(subset)
        exec_stats = execution[method]

        rows.append(
            {
                "method": method,
                "attempted_count": exec_stats["attempted_count"],
                "present_result_count": exec_stats["present_result_count"],
                "missing_result_count": exec_stats["missing_result_count"],
                "saved_timeout_count": exec_stats["saved_timeout_count"],
                "trace_count": len(subset),
                "semantic_evaluable_count": metrics["semantic_evaluable_count"],
                "after_em_accuracy": metrics["after_em_accuracy"],
                "after_mean_token_f1": metrics["after_mean_token_f1"],
                "after_semantic_accuracy": metrics["after_semantic_accuracy"],
                "semantic_recovery_rate": metrics["semantic_recovery_rate"],
                "semantic_regression_rate": metrics["semantic_regression_rate"],
                "fully_repaired_rate": metrics["fully_repaired_rate"],
                "regression_detected_rate": metrics["regression_detected_rate"],
            }
        )

    return {
        "condition": condition,
        "attempted_trace_count": attempted_count,
        "common_present_trace_count": len(common_present_ids),
        # Keep this key for compatibility, but make it the true paired count.
        "common_trace_count": len(paired_ids),
        "paired_common_trace_count": len(paired_ids),
        "jointly_unevaluable_count": (
            len(common_present_ids) - len(paired_ids)
        ),
        "execution": execution,
        "methods": rows,
    }


def save_summary_csv(summary: Dict[str, Any], output_path: Path) -> None:
    fields = [
        "experiment",
        "repair_mode",
        "total",
        "saved_timeout_count",
        "semantic_evaluable_count",
        "semantic_indeterminate_count",
        "before_em_accuracy",
        "after_em_accuracy",
        "em_gain",
        "before_mean_token_f1",
        "after_mean_token_f1",
        "token_f1_gain",
        "before_semantic_accuracy",
        "after_semantic_accuracy",
        "semantic_gain",
        "semantic_recovery_rate",
        "semantic_regression_rate",
        "fully_repaired_rate",
        "remaining_M_count",
        "remaining_D_count",
        "remaining_L_count",
        "new_failure_count",
        "regression_detected_rate",
        "L_rewrite_applied_count",
        "safe_kept_step_count",
        "safe_reverted_step_count",
        "safe_planner_technical_error_count",
        "safe_step_technical_error_count",
        "generation_failure_count",
        "semantic_judge_error_count",
    ]

    rows = []
    for experiment, metrics in summary.items():
        rows.append(
            {
                field: experiment if field == "experiment" else metrics.get(field, "")
                for field in fields
            }
        )

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_method_matrix(
    comparisons: Dict[str, Any],
    output_path: Path,
) -> None:
    fields = [
        "condition",
        "method",
        "attempted_count",
        "present_result_count",
        "missing_result_count",
        "saved_timeout_count",
        "trace_count",
        "semantic_evaluable_count",
        "after_em_accuracy",
        "after_mean_token_f1",
        "after_semantic_accuracy",
        "semantic_recovery_rate",
        "semantic_regression_rate",
        "fully_repaired_rate",
        "regression_detected_rate",
    ]

    rows = []
    for condition, comparison in comparisons.items():
        for method in comparison.get("methods", []):
            rows.append({"condition": condition, **method})

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    repair_results = {
        name: load_results(path)
        for name, path in EXPERIMENTS.items()
        if Path(path).exists()
    }

    if not repair_results:
        raise FileNotFoundError("No repair result files were found.")

    # Individual experiment summaries use each experiment's own completed file.
    summary = {
        experiment: calculate_metrics(traces)
        for experiment, traces in repair_results.items()
    }

    # Cross-method matrices use exactly the same paired, jointly evaluable
    # trace IDs across every method included for that condition.
    comparisons = {}

    for condition, core_group in METHOD_GROUPS.items():
        failure_path = Path(FAILURE_RESULTS[condition])

        if not failure_path.exists():
            continue

        if not all(
            experiment in repair_results
            for experiment in core_group.values()
        ):
            continue

        # Preserve the existing fixed/reverse/Safe comparison when B2/B3 have
        # not been run yet. Once their result files exist, add them to the same
        # paired matrix so every displayed row uses the same common trace IDs.
        optional_group = OPTIONAL_BASELINE_METHODS.get(condition, {})
        group = {
            method: experiment
            for method, experiment in optional_group.items()
            if experiment in repair_results
        }
        group.update(core_group)

        comparisons[condition] = method_comparison(
            condition=condition,
            group=group,
            repair_results=repair_results,
            failure_results=load_results(str(failure_path)),
        )

    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "repair_summary.json"
    csv_path = output_dir / "repair_summary.csv"
    matrix_path = output_dir / "repair_method_matrix.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "experiments": summary,
                "method_comparisons": comparisons,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    save_summary_csv(summary, csv_path)

    # This call was missing in the previous version.
    save_method_matrix(comparisons, matrix_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {matrix_path}")


if __name__ == "__main__":
    main()