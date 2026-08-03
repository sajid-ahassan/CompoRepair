import json
import os


def load_results(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(results):

    total = len(results)

    exact = sum(1 for item in results if item["evaluation"]["exact_match"])

    semantic = sum(1 for item in results if item["evaluation"]["semantic_correct"])

    return {
        "total": total,
        "exact_match_accuracy": exact / total if total else 0,
        "semantic_accuracy": semantic / total if total else 0,
    }


def main():

    experiments = {
        "baseline": "data/processed/pilot_baseline_results.json",
        "M": "data/results/single/missing_evidence_results.json",
        "D": "data/results/single/distractor_results.json",
        "G": "data/results/single/reasoning_results.json",
        "M_D": "data/results/compound/M_D_results.json",
        "M_G": "data/results/compound/M_G_results.json",
        "D_G": "data/results/compound/D_G_results.json",
    }

    summary = {}

    for name, path in experiments.items():

        results = load_results(path)

        summary[name] = calculate_metrics(results)

    baseline_em = summary["baseline"]["exact_match_accuracy"]
    baseline_sem = summary["baseline"]["semantic_accuracy"]

    for name in summary:

        if name == "baseline":
            continue

        summary[name]["em_drop"] = baseline_em - summary[name]["exact_match_accuracy"]

        summary[name]["semantic_drop"] = (
            baseline_sem - summary[name]["semantic_accuracy"]
        )

    os.makedirs("data/analysis", exist_ok=True)

    with open("data/analysis/failure_summary.json", "w", encoding="utf-8") as f:

        json.dump(summary, f, indent=2)

    print("Failure analysis completed.")


if __name__ == "__main__":
    main()
