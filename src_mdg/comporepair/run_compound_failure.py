import json
import os
from copy import deepcopy

from .pipeline.baseline_rag import (
    evaluate_prediction,
    generate_answer,
    generate_plan_final_answer,
    generate_planned_answer,
    model_manifest,
)

EXPERIMENTS = [
    (
        ["M", "D"],
        "data/failures/compound/M_D.json",
        "data/results/compound/M_D_results.json",
    ),
    (
        ["M", "G"],
        "data/failures/compound/M_G.json",
        "data/results/compound/M_G_results.json",
    ),
    (
        ["D", "G"],
        "data/failures/compound/D_G.json",
        "data/results/compound/D_G_results.json",
    ),
]


def run_compound_failure(
    input_path: str,
    output_path: str,
    failures,
) -> None:
    with open(input_path, "r", encoding="utf-8") as file:
        traces = json.load(file)

    label = "_".join(failures)
    results = []
    for index, source_trace in enumerate(traces, start=1):
        trace = deepcopy(source_trace)

        clean_control = None
        if "G" in trace["true_failures"]:
            generation = generate_planned_answer(
                trace["question"],
                trace["retrieval_events"],
                plan=trace["corrupted_reasoning_plan"],
            )
            clean_control = generate_plan_final_answer(
                trace["question"],
                trace["reasoning_plan"],
                generation.get("reasoning_outputs", {}),
            )
        else:
            generation = generate_answer(
                trace["question"],
                trace["retrieval_events"],
            )

        answer = generation["text"]
        evaluation = evaluate_prediction(
            trace["question"],
            answer,
            trace["canonical_answer"],
            generation["generation_failed"],
        )

        trace["experiment_stage"] = f"compound_failure_result_{label}"
        trace["failure_answer"] = answer
        trace["final_answer"] = answer
        trace["failure_evaluation"] = evaluation
        trace["evaluation"] = evaluation
        if "G" in trace["true_failures"]:
            trace["reasoning_outputs"] = generation["reasoning_outputs"]
            control_answer = str(clean_control.get("text", "") or "")
            trace["g_clean_control_answer"] = control_answer
            trace["g_clean_control_evaluation"] = evaluate_prediction(
                trace["question"],
                control_answer,
                trace["canonical_answer"],
                clean_control.get("generation_failed", False),
            )
            trace["g_clean_control_reused_reasoning_outputs"] = True
            trace["g_clean_control_latency_ms"] = int(
                clean_control.get("latency_ms", 0)
            )
            trace["g_clean_control_token_usage"] = clean_control.get(
                "token_usage", {}
            )
        trace["answer_claims"] = [
            {
                "claim": answer,
                "source": f"compound_failure_{label}",
                "evidence_ids": [
                    item.get("passage_id", "")
                    for item in trace.get("retrieval_events", [])
                ],
            }
        ]
        trace["model_manifest"] = model_manifest()
        prompt_hashes = dict(trace.get("prompt_hashes", {}))
        prompt_hashes[f"failure_{label}"] = generation["prompt_hash"]
        if clean_control is not None:
            prompt_hashes["g_clean_control"] = clean_control.get(
                "prompt_hash", ""
            )
        trace["prompt_hashes"] = prompt_hashes
        trace["latency_ms"] = int(generation["latency_ms"])
        trace["token_usage"] = generation["token_usage"]

        results.append(trace)
        print(f"{label} processed trace {index}/{len(traces)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


def main():
    for failures, input_path, output_path in EXPERIMENTS:
        run_compound_failure(input_path, output_path, failures)


if __name__ == "__main__":
    main()