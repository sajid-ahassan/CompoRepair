import json
from pathlib import Path

EXPERIMENTS = {
    "baseline": "data/processed/pilot_baseline_results.json",
    "M": "data/results/single/missing_evidence_results.json",
    "D": "data/results/single/distractor_results.json",
    "G": "data/results/single/reasoning_results.json",
    "M_D": "data/results/compound/M_D_results.json",
    "M_G": "data/results/compound/M_G_results.json",
    "D_G": "data/results/compound/D_G_results.json",
}


def load_results(path):
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(results):
    total = len(results)

    exact_count = sum(
        bool(item.get("evaluation", {}).get("exact_match", False)) for item in results
    )

    semantic_count = sum(
        bool(item.get("evaluation", {}).get("semantic_correct", False))
        for item in results
    )

    return {
        "total": total,
        "exact_match_count": exact_count,
        "semantic_correct_count": semantic_count,
        "exact_match_accuracy": exact_count / total if total else 0,
        "semantic_accuracy": semantic_count / total if total else 0,
    }


def calculate_transitions(baseline_results, failure_results):
    """
    Compare each failure result with its matching baseline trace.
    """

    baseline_map = {trace["trace_id"]: trace for trace in baseline_results}

    transitions = {
        "correct_to_wrong": 0,
        "wrong_to_correct": 0,
        "stayed_correct": 0,
        "stayed_wrong": 0,
        "matched_traces": 0,
    }

    for failure_trace in failure_results:
        trace_id = failure_trace.get("trace_id")

        if trace_id not in baseline_map:
            continue

        baseline_trace = baseline_map[trace_id]

        baseline_correct = bool(
            baseline_trace.get("evaluation", {}).get(
                "semantic_correct",
                False,
            )
        )

        failure_correct = bool(
            failure_trace.get("evaluation", {}).get(
                "semantic_correct",
                False,
            )
        )

        transitions["matched_traces"] += 1

        if baseline_correct and not failure_correct:
            transitions["correct_to_wrong"] += 1

        elif not baseline_correct and failure_correct:
            transitions["wrong_to_correct"] += 1

        elif baseline_correct and failure_correct:
            transitions["stayed_correct"] += 1

        else:
            transitions["stayed_wrong"] += 1

    return transitions


def add_compound_interaction(summary):
    """
    Measure whether a compound failure causes additional degradation
    beyond its strongest individual component.
    """

    compound_pairs = {
        "M_D": ("M", "D"),
        "M_G": ("M", "G"),
        "D_G": ("D", "G"),
    }

    for compound, singles in compound_pairs.items():
        first, second = singles

        strongest_single_semantic_drop = max(
            summary[first]["semantic_drop"],
            summary[second]["semantic_drop"],
        )

        strongest_single_em_drop = max(
            summary[first]["em_drop"],
            summary[second]["em_drop"],
        )

        summary[compound]["extra_semantic_drop_vs_strongest_single"] = (
            summary[compound]["semantic_drop"] - strongest_single_semantic_drop
        )

        summary[compound]["extra_em_drop_vs_strongest_single"] = (
            summary[compound]["em_drop"] - strongest_single_em_drop
        )


def main():
    results_by_experiment = {
        name: load_results(path) for name, path in EXPERIMENTS.items()
    }

    summary = {
        name: calculate_metrics(results)
        for name, results in results_by_experiment.items()
    }

    baseline_em = summary["baseline"]["exact_match_accuracy"]
    baseline_semantic = summary["baseline"]["semantic_accuracy"]

    baseline_results = results_by_experiment["baseline"]

    for name in summary:
        if name == "baseline":
            continue

        summary[name]["em_drop"] = baseline_em - summary[name]["exact_match_accuracy"]

        summary[name]["semantic_drop"] = (
            baseline_semantic - summary[name]["semantic_accuracy"]
        )

        summary[name]["transitions"] = calculate_transitions(
            baseline_results,
            results_by_experiment[name],
        )

    add_compound_interaction(summary)

    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "failure_summary.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Failure analysis completed: {output_path}")


if __name__ == "__main__":
    main()
