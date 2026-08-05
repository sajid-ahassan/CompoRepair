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
        "M",
        "data/failures/single/missing_evidence.json",
        "data/results/single/missing_evidence_results.json",
    ),
    (
        "D",
        "data/failures/single/distractor.json",
        "data/results/single/distractor_results.json",
    ),
    (
        "G",
        "data/failures/single/reasoning.json",
        "data/results/single/reasoning_results.json",
    ),
]


def run_single_failure(input_path: str, output_path: str, failure_type: str) -> None:
    with open(input_path, "r", encoding="utf-8") as file:
        traces = json.load(file)

    results = []
    for index, source_trace in enumerate(traces, start=1):
        trace = deepcopy(source_trace)

        clean_control = None
        if failure_type == "G":
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

        trace["experiment_stage"] = f"single_failure_result_{failure_type}"
        trace["failure_answer"] = answer
        trace["final_answer"] = answer
        trace["failure_evaluation"] = evaluation
        trace["evaluation"] = evaluation
        if failure_type == "G":
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
                "source": f"single_failure_{failure_type}",
                "evidence_ids": [
                    item.get("passage_id", "")
                    for item in trace.get("retrieval_events", [])
                ],
            }
        ]
        trace["model_manifest"] = model_manifest()
        prompt_hashes = dict(trace.get("prompt_hashes", {}))
        prompt_hashes[f"failure_{failure_type}"] = generation["prompt_hash"]
        if clean_control is not None:
            prompt_hashes["g_clean_control"] = clean_control.get(
                "prompt_hash", ""
            )
        trace["prompt_hashes"] = prompt_hashes
        trace["latency_ms"] = int(generation["latency_ms"])
        trace["token_usage"] = generation["token_usage"]

        results.append(trace)
        print(f"{failure_type} processed trace {index}/{len(traces)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


def main():
    for failure_type, input_path, output_path in EXPERIMENTS:
        run_single_failure(input_path, output_path, failure_type)


if __name__ == "__main__":
    main()