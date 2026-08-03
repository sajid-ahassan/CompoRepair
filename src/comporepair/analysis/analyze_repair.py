import csv
import json
from pathlib import Path

EXPERIMENTS = {
    "M": "data/repaired_result/M_repair_results.json",
    "D": "data/repaired_result/D_repair_results.json",
    "G": "data/repaired_result/G_repair_results.json",
    "M_D": "data/repaired_result/M_D_repair_results.json",
    "M_G": "data/repaired_result/M_G_repair_results.json",
    "D_G": "data/repaired_result/D_G_repair_results.json",
}


def load_results(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_evaluation(trace, stage):
    """
    Supports:
    repair_evaluation[stage]["evaluation"]
    and direct evaluation fields.
    """

    stage_data = trace.get(
        "repair_evaluation",
        {},
    ).get(stage, {})

    return stage_data.get(
        "evaluation",
        stage_data,
    )


def calculate_metrics(traces):
    total = len(traces)

    before_em = 0
    after_em = 0
    before_semantic = 0
    after_semantic = 0

    semantic_recovered = 0
    semantic_regressed = 0
    semantic_stayed_correct = 0
    semantic_stayed_wrong = 0

    em_recovered = 0
    em_regressed = 0

    for trace in traces:
        before = get_evaluation(trace, "before")
        after = get_evaluation(trace, "after")

        before_exact = bool(before.get("exact_match", False))
        after_exact = bool(after.get("exact_match", False))

        before_sem = bool(before.get("semantic_correct", False))
        after_sem = bool(after.get("semantic_correct", False))

        before_em += before_exact
        after_em += after_exact
        before_semantic += before_sem
        after_semantic += after_sem

        if not before_exact and after_exact:
            em_recovered += 1

        if before_exact and not after_exact:
            em_regressed += 1

        if not before_sem and after_sem:
            semantic_recovered += 1

        elif before_sem and not after_sem:
            semantic_regressed += 1

        elif before_sem and after_sem:
            semantic_stayed_correct += 1

        else:
            semantic_stayed_wrong += 1

    before_wrong = total - before_semantic
    before_correct = before_semantic

    return {
        "total": total,
        "before_em_accuracy": (before_em / total if total else 0),
        "after_em_accuracy": (after_em / total if total else 0),
        "em_gain": ((after_em - before_em) / total if total else 0),
        "before_semantic_accuracy": (before_semantic / total if total else 0),
        "after_semantic_accuracy": (after_semantic / total if total else 0),
        "semantic_gain": ((after_semantic - before_semantic) / total if total else 0),
        "em_recovered_count": em_recovered,
        "em_regressed_count": em_regressed,
        "semantic_recovered_count": semantic_recovered,
        "semantic_regressed_count": semantic_regressed,
        "semantic_stayed_correct": semantic_stayed_correct,
        "semantic_stayed_wrong": semantic_stayed_wrong,
        "semantic_recovery_rate": (
            semantic_recovered / before_wrong if before_wrong else 0
        ),
        "semantic_regression_rate": (
            semantic_regressed / before_correct if before_correct else 0
        ),
    }


def save_csv(summary, output_path):
    rows = []

    for experiment, metrics in summary.items():
        rows.append(
            {
                "experiment": experiment,
                **metrics,
            }
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    output_dir = Path("data/analysis")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {}

    for experiment, path in EXPERIMENTS.items():
        if not Path(path).exists():
            print(f"Skipped missing file: {path}")
            continue

        traces = load_results(path)

        summary[experiment] = calculate_metrics(traces)

    if not summary:
        raise FileNotFoundError("No repair result files were found.")

    json_path = output_dir / "repair_summary.json"
    csv_path = output_dir / "repair_summary.csv"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # save_csv(summary, csv_path)

    print(f"Saved: {json_path}")
    # print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
