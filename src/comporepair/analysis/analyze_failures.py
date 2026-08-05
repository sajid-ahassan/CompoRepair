import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42

EXPERIMENTS = {
    "baseline": "data/processed/pilot_baseline_results.json",
    "M": "data/results/single/missing_evidence_results.json",
    "D": "data/results/single/distractor_results.json",
    "G": "data/results/single/reasoning_results.json",
    "M_D": "data/results/compound/M_D_results.json",
    "M_G": "data/results/compound/M_G_results.json",
    "D_G": "data/results/compound/D_G_results.json",
}

SKIPPED = {
    "M": "data/failures/single/missing_evidence_skipped.json",
    "D": "data/failures/single/distractor_skipped.json",
    "G": "data/failures/single/reasoning_skipped.json",
    "M_D": "data/failures/compound/M_D_skipped.json",
    "M_G": "data/failures/compound/M_G_skipped.json",
    "D_G": "data/failures/compound/D_G_skipped.json",
}

COMPOUNDS = {
    "M_D": ("M", "D"),
    "M_G": ("M", "G"),
    "D_G": ("D", "G"),
}


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _evaluation(trace: Dict[str, Any], stage: str) -> Dict[str, Any]:
    if stage == "baseline":
        return trace.get("baseline_evaluation") or trace.get("evaluation", {})
    return trace.get("failure_evaluation") or trace.get("evaluation", {})


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def bootstrap_ci(values: List[float]) -> List[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(sample) / len(sample))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def calculate_metrics(traces: List[Dict[str, Any]], stage: str) -> Dict[str, Any]:
    evaluations = [_evaluation(trace, stage) for trace in traces]
    total = len(evaluations)
    em_values = [float(bool(item.get("exact_match", False))) for item in evaluations]
    f1_values = [float(item.get("token_f1", 0.0)) for item in evaluations]
    semantic_values = [
        float(bool(item.get("semantic_correct", False))) for item in evaluations
    ]
    generation_failures = sum(
        bool(item.get("generation_failed", False)) for item in evaluations
    )
    judge_errors = sum(
        bool(item.get("semantic_judge_error", False)) for item in evaluations
    )

    return {
        "total": total,
        "exact_match_count": int(sum(em_values)),
        "semantic_correct_count": int(sum(semantic_values)),
        "exact_match_accuracy": sum(em_values) / total if total else 0.0,
        "mean_token_f1": sum(f1_values) / total if total else 0.0,
        "semantic_accuracy": sum(semantic_values) / total if total else 0.0,
        "exact_match_ci95": bootstrap_ci(em_values),
        "token_f1_ci95": bootstrap_ci(f1_values),
        "semantic_ci95": bootstrap_ci(semantic_values),
        "generation_failure_count": generation_failures,
        "generation_failure_rate": generation_failures / total if total else 0.0,
        "semantic_judge_error_count": judge_errors,
        "semantic_judge_error_rate": judge_errors / total if total else 0.0,
    }


def _trace_map(traces: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(trace.get("trace_id", "")): trace for trace in traces}


def transitions(
    baseline: List[Dict[str, Any]],
    failure: List[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_map = _trace_map(baseline)
    counts = {
        "correct_to_wrong": 0,
        "wrong_to_correct": 0,
        "stayed_correct": 0,
        "stayed_wrong": 0,
        "matched_traces": 0,
    }
    effectiveness_values = []

    for trace in failure:
        baseline_trace = baseline_map.get(str(trace.get("trace_id", "")))
        if not baseline_trace:
            continue
        before = bool(_evaluation(baseline_trace, "baseline").get("semantic_correct", False))
        after = bool(_evaluation(trace, "failure").get("semantic_correct", False))
        counts["matched_traces"] += 1

        if before and not after:
            counts["correct_to_wrong"] += 1
        elif not before and after:
            counts["wrong_to_correct"] += 1
        elif before and after:
            counts["stayed_correct"] += 1
        else:
            counts["stayed_wrong"] += 1

        if before:
            effectiveness_values.append(float(not after))

    counts["failure_effectiveness"] = (
        sum(effectiveness_values) / len(effectiveness_values)
        if effectiveness_values
        else 0.0
    )
    counts["failure_effectiveness_ci95"] = bootstrap_ci(effectiveness_values)
    return counts


def g_dependency_control_metrics(
    traces: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Measure the isolated effect of removing one G dependency.

    The clean and corrupted final steps reuse the same intermediate reasoning
    outputs, so this comparison is not confounded by different evidence-step
    generations.
    """
    valid_controls = 0
    clean_correct = 0
    dependency_effective = 0
    stayed_correct = 0
    control_wrong = 0
    indeterminate = 0
    effectiveness_values = []

    for trace in traces:
        control = trace.get("g_clean_control_evaluation", {})
        corrupted = _evaluation(trace, "failure")

        technical_error = bool(
            not control
            or control.get("generation_failed", False)
            or control.get("semantic_judge_error", False)
            or corrupted.get("generation_failed", False)
            or corrupted.get("semantic_judge_error", False)
        )
        if technical_error:
            indeterminate += 1
            continue

        valid_controls += 1
        control_is_correct = bool(control.get("semantic_correct", False))
        corrupted_is_correct = bool(
            corrupted.get("semantic_correct", False)
        )

        if control_is_correct:
            clean_correct += 1
            changed_to_wrong = not corrupted_is_correct
            dependency_effective += int(changed_to_wrong)
            stayed_correct += int(corrupted_is_correct)
            effectiveness_values.append(float(changed_to_wrong))
        else:
            control_wrong += 1

    return {
        "G_clean_control_valid_count": valid_controls,
        "G_clean_control_semantic_correct_count": clean_correct,
        "G_clean_control_semantic_accuracy": (
            clean_correct / valid_controls if valid_controls else 0.0
        ),
        "G_dependency_effective_count": dependency_effective,
        "G_dependency_stayed_correct_count": stayed_correct,
        "G_clean_control_wrong_count": control_wrong,
        "G_dependency_comparison_indeterminate_count": indeterminate,
        "G_dependency_effectiveness": (
            dependency_effective / clean_correct if clean_correct else 0.0
        ),
        "G_dependency_effectiveness_ci95": bootstrap_ci(
            effectiveness_values
        ),
    }


def _evidence_signature(trace: Dict[str, Any]):
    return tuple(
        (
            str(item.get("passage_id", "")),
            str(item.get("title", "")),
            str(item.get("text", "")),
        )
        for item in trace.get("retrieval_events", [])
    )


def _g_plan_corruption_valid(trace: Dict[str, Any]) -> bool:
    clean_plan = trace.get("reasoning_plan", {})
    corrupted_plan = trace.get("corrupted_reasoning_plan", {})

    clean_steps = clean_plan.get("steps", []) if isinstance(clean_plan, dict) else []
    corrupted_steps = (
        corrupted_plan.get("steps", [])
        if isinstance(corrupted_plan, dict)
        else []
    )

    if not clean_steps or len(clean_steps) != len(corrupted_steps):
        return False

    for index, (clean_step, corrupted_step) in enumerate(
        zip(clean_steps, corrupted_steps)
    ):
        if not isinstance(clean_step, dict) or not isinstance(corrupted_step, dict):
            return False
        if clean_step.get("id") != corrupted_step.get("id"):
            return False
        if clean_step.get("task") != corrupted_step.get("task"):
            return False
        if index < len(clean_steps) - 1:
            if clean_step.get("depends_on", []) != corrupted_step.get(
                "depends_on", []
            ):
                return False

    clean_final = clean_steps[-1]
    corrupted_final = corrupted_steps[-1]
    clean_dependencies = clean_final.get("depends_on", [])
    corrupted_dependencies = corrupted_final.get("depends_on", [])

    if not isinstance(clean_dependencies, list):
        return False
    if not isinstance(corrupted_dependencies, list):
        return False
    if len(clean_dependencies) < 2:
        return False
    if len(corrupted_dependencies) != len(clean_dependencies) - 1:
        return False

    removed_dependencies = [
        dependency
        for dependency in clean_dependencies
        if dependency not in corrupted_dependencies
    ]
    added_dependencies = [
        dependency
        for dependency in corrupted_dependencies
        if dependency not in clean_dependencies
    ]

    if len(removed_dependencies) != 1 or added_dependencies:
        return False

    expected_corrupted_dependencies = [
        dependency
        for dependency in clean_dependencies
        if dependency != removed_dependencies[0]
    ]
    if corrupted_dependencies != expected_corrupted_dependencies:
        return False

    removed_dependency = str(removed_dependencies[0])
    corrupted_step_id = str(clean_final.get("id", ""))

    history_entry = next(
        (
            item
            for item in trace.get("failure_history", [])
            if item.get("failure") == "G"
        ),
        {},
    )

    return (
        str(trace.get("g_removed_dependency", "")) == removed_dependency
        and str(trace.get("g_corrupted_step_id", "")) == corrupted_step_id
        and history_entry.get("intervention_version")
        == "g_dependency_removal_v1"
        and history_entry.get("failure_type")
        == "reasoning_dependency_removed"
        and str(history_entry.get("removed_dependency", ""))
        == removed_dependency
        and str(history_entry.get("corrupted_step_id", ""))
        == corrupted_step_id
    )


def injection_validity(
    name: str,
    traces: List[Dict[str, Any]],
    results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    valid = 0
    invalid = 0
    g_reference_name = {"G": "baseline", "M_G": "M", "D_G": "D"}.get(name)
    g_reference_map = (
        _trace_map(results[g_reference_name]) if g_reference_name else {}
    )

    for trace in traces:
        events = trace.get("retrieval_events", [])
        ids = {str(item.get("passage_id", "")) for item in events}
        history = trace.get("failure_history", [])
        labels = [item.get("failure") for item in history]
        checks = []

        if "M" in name.split("_"):
            entries = [item for item in history if item.get("failure") == "M"]
            checks.append(
                len(entries) == 1
                and all(pid not in ids for pid in entries[0].get("removed_passage_ids", []))
            )
        if "D" in name.split("_"):
            entries = [item for item in history if item.get("failure") == "D"]
            checks.append(
                len(entries) == 1
                and entries[0].get("distractor_passage_id") in ids
                and entries[0].get("source_passage_id") in ids
                and len(events) == (5 if "M" in name.split("_") else 6)
            )
        if "G" in name.split("_"):
            entries = [item for item in history if item.get("failure") == "G"]
            reference_trace = g_reference_map.get(str(trace.get("trace_id", "")))
            checks.append(
                len(entries) == 1
                and entries[0].get("evidence_modified") is False
                and reference_trace is not None
                and _evidence_signature(trace) == _evidence_signature(reference_trace)
                and _g_plan_corruption_valid(trace)
                and isinstance(trace.get("g_clean_control_evaluation"), dict)
                and bool(trace.get("g_clean_control_evaluation"))
                and trace.get("g_clean_control_reused_reasoning_outputs") is True
            )

        expected = name.split("_")
        checks.append(all(labels.count(label) == 1 for label in expected))
        if all(checks):
            valid += 1
        else:
            invalid += 1

    return {
        "valid_injection_count": valid,
        "invalid_injection_count": invalid,
        "injection_validity_rate": valid / len(traces) if traces else 0.0,
    }


def skip_summary(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        return {"skipped_count": 0, "skip_reason_counts": {}}
    items = load_json(path)
    reasons = Counter(str(item.get("reason", "unknown")) for item in items)
    return {
        "skipped_count": len(items),
        "skip_reason_counts": dict(reasons),
    }


def _subset(traces: List[Dict[str, Any]], ids: set) -> List[Dict[str, Any]]:
    return [trace for trace in traces if str(trace.get("trace_id", "")) in ids]


def _metric_drop(
    baseline: List[Dict[str, Any]],
    condition: List[Dict[str, Any]],
) -> Dict[str, float]:
    base = calculate_metrics(baseline, "baseline")
    result = calculate_metrics(condition, "failure")
    return {
        "em": base["exact_match_accuracy"] - result["exact_match_accuracy"],
        "token_f1": base["mean_token_f1"] - result["mean_token_f1"],
        "semantic": base["semantic_accuracy"] - result["semantic_accuracy"],
    }


def _compound_extra_ci(
    baseline: List[Dict[str, Any]],
    first: List[Dict[str, Any]],
    second: List[Dict[str, Any]],
    compound: List[Dict[str, Any]],
    metric_key: str,
) -> List[float]:
    baseline_map = _trace_map(baseline)
    first_map = _trace_map(first)
    second_map = _trace_map(second)
    compound_map = _trace_map(compound)
    ids = sorted(set(baseline_map) & set(first_map) & set(second_map) & set(compound_map))
    if not ids:
        return [0.0, 0.0]

    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]

        def mean_for(mapping, stage):
            values = []
            for trace_id in sampled_ids:
                evaluation = _evaluation(mapping[trace_id], stage)
                if metric_key == "exact_match":
                    values.append(float(bool(evaluation.get("exact_match", False))))
                elif metric_key == "token_f1":
                    values.append(float(evaluation.get("token_f1", 0.0)))
                else:
                    values.append(float(bool(evaluation.get("semantic_correct", False))))
            return sum(values) / len(values)

        base_value = mean_for(baseline_map, "baseline")
        first_drop = base_value - mean_for(first_map, "failure")
        second_drop = base_value - mean_for(second_map, "failure")
        compound_drop = base_value - mean_for(compound_map, "failure")
        estimates.append(compound_drop - max(first_drop, second_drop))

    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def compound_interaction(
    name: str,
    results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    first, second = COMPOUNDS[name]
    common_ids = (
        set(_trace_map(results["baseline"]))
        & set(_trace_map(results[first]))
        & set(_trace_map(results[second]))
        & set(_trace_map(results[name]))
    )
    if not common_ids:
        return {"common_trace_count": 0}

    baseline = _subset(results["baseline"], common_ids)
    first_traces = _subset(results[first], common_ids)
    second_traces = _subset(results[second], common_ids)
    compound_traces = _subset(results[name], common_ids)

    first_drop = _metric_drop(baseline, first_traces)
    second_drop = _metric_drop(baseline, second_traces)
    compound_drop = _metric_drop(baseline, compound_traces)

    return {
        "common_trace_count": len(common_ids),
        "single_drops": {first: first_drop, second: second_drop},
        "compound_drop": compound_drop,
        "extra_em_drop_vs_strongest_single": compound_drop["em"]
        - max(first_drop["em"], second_drop["em"]),
        "extra_em_drop_ci95": _compound_extra_ci(
            baseline, first_traces, second_traces, compound_traces, "exact_match"
        ),
        "extra_token_f1_drop_vs_strongest_single": compound_drop["token_f1"]
        - max(first_drop["token_f1"], second_drop["token_f1"]),
        "extra_token_f1_drop_ci95": _compound_extra_ci(
            baseline, first_traces, second_traces, compound_traces, "token_f1"
        ),
        "extra_semantic_drop_vs_strongest_single": compound_drop["semantic"]
        - max(first_drop["semantic"], second_drop["semantic"]),
        "extra_semantic_drop_ci95": _compound_extra_ci(
            baseline, first_traces, second_traces, compound_traces, "semantic_correct"
        ),
    }


def save_csv(summary: Dict[str, Any], path: Path) -> None:
    fields = [
        "condition",
        "total",
        "eligible_count",
        "skipped_count",
        "exact_match_accuracy",
        "mean_token_f1",
        "semantic_accuracy",
        "em_drop",
        "token_f1_drop",
        "semantic_drop",
        "correct_to_wrong",
        "wrong_to_correct",
        "stayed_correct",
        "stayed_wrong",
        "failure_effectiveness",
        "generation_failure_rate",
        "semantic_judge_error_rate",
        "invalid_injection_count",
        "G_clean_control_valid_count",
        "G_clean_control_semantic_accuracy",
        "G_dependency_effective_count",
        "G_dependency_effectiveness",
        "G_dependency_comparison_indeterminate_count",
    ]
    rows = []
    for name, item in summary.items():
        if name == "baseline":
            rows.append(
                {
                    "condition": name,
                    **{field: item.get(field, "") for field in fields[1:]},
                }
            )
            continue
        transition = item.get("transitions", {})
        rows.append(
            {
                "condition": name,
                "total": item.get("total", 0),
                "eligible_count": item.get("eligible_count", 0),
                "skipped_count": item.get("skipped_count", 0),
                "exact_match_accuracy": item.get("exact_match_accuracy", 0.0),
                "mean_token_f1": item.get("mean_token_f1", 0.0),
                "semantic_accuracy": item.get("semantic_accuracy", 0.0),
                "em_drop": item.get("em_drop", 0.0),
                "token_f1_drop": item.get("token_f1_drop", 0.0),
                "semantic_drop": item.get("semantic_drop", 0.0),
                "correct_to_wrong": transition.get("correct_to_wrong", 0),
                "wrong_to_correct": transition.get("wrong_to_correct", 0),
                "stayed_correct": transition.get("stayed_correct", 0),
                "stayed_wrong": transition.get("stayed_wrong", 0),
                "failure_effectiveness": transition.get("failure_effectiveness", 0.0),
                "generation_failure_rate": item.get("generation_failure_rate", 0.0),
                "semantic_judge_error_rate": item.get("semantic_judge_error_rate", 0.0),
                "invalid_injection_count": item.get("invalid_injection_count", 0),
                "G_clean_control_valid_count": item.get(
                    "G_clean_control_valid_count", ""
                ),
                "G_clean_control_semantic_accuracy": item.get(
                    "G_clean_control_semantic_accuracy", ""
                ),
                "G_dependency_effective_count": item.get(
                    "G_dependency_effective_count", ""
                ),
                "G_dependency_effectiveness": item.get(
                    "G_dependency_effectiveness", ""
                ),
                "G_dependency_comparison_indeterminate_count": item.get(
                    "G_dependency_comparison_indeterminate_count", ""
                ),
            }
        )

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    results = {name: load_json(path) for name, path in EXPERIMENTS.items()}
    baseline = results["baseline"]
    baseline_metrics = calculate_metrics(baseline, "baseline")
    baseline_metrics["duplicate_trace_count"] = len(baseline) - len(_trace_map(baseline))
    summary = {"baseline": baseline_metrics}

    for name in ("M", "D", "G", "M_D", "M_G", "D_G"):
        traces = results[name]
        ids = set(_trace_map(traces))
        matched_baseline = _subset(baseline, ids)
        condition_metrics = calculate_metrics(traces, "failure")
        base_metrics = calculate_metrics(matched_baseline, "baseline")
        condition_metrics.update(
            {
                "eligible_count": len(traces),
                "duplicate_trace_count": len(traces) - len(_trace_map(traces)),
                **skip_summary(SKIPPED[name]),
                "unmatched_trace_count": len(traces) - len(matched_baseline),
                "matched_baseline": base_metrics,
                "em_drop": base_metrics["exact_match_accuracy"]
                - condition_metrics["exact_match_accuracy"],
                "token_f1_drop": base_metrics["mean_token_f1"]
                - condition_metrics["mean_token_f1"],
                "semantic_drop": base_metrics["semantic_accuracy"]
                - condition_metrics["semantic_accuracy"],
                "transitions": transitions(baseline, traces),
                **injection_validity(name, traces, results),
            }
        )
        if "G" in name.split("_"):
            condition_metrics.update(
                g_dependency_control_metrics(traces)
            )
        if name in COMPOUNDS:
            condition_metrics["compound_interaction"] = compound_interaction(
                name, results
            )
        summary[name] = condition_metrics

    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "failure_summary.json"
    csv_path = output_dir / "failure_summary.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    save_csv(summary, csv_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()