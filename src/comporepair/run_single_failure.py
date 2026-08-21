import json
import multiprocessing as mp
import os
from copy import deepcopy

from .pipeline.baseline_rag import (
    evaluate_prediction,
    generate_answer,
    model_manifest,
)

TRACE_TIMEOUT_SECONDS = 180.0


def _timeout_output_path(output_path: str) -> str:
    name = os.path.splitext(os.path.basename(output_path))[0] + "_timeouts.json"
    return os.path.join("data", "timeouts", "single", name)

EXPERIMENTS = [
    (
        "M",
        "data/final_test_inputs/single/missing_evidence.json",
        "data/results/single/missing_evidence_results.json",
    ),
    (
        "D",
        "data/final_test_inputs/single/distractor.json",
        "data/results/single/distractor_results.json",
    ),
    (
        "L",
        "data/final_test_inputs/single/linkage.json",
        "data/results/single/linkage_results.json",
    ),
]


def _single_failure_call(trace):
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
    return generation, answer, evaluation


def run_single_failure(input_path: str, output_path: str, failure_type: str) -> None:
    with open(input_path, "r", encoding="utf-8") as file:
        traces = json.load(file)

    results = []
    timeouts = []
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=1)

    try:
        for index, source_trace in enumerate(traces, start=1):
            trace = deepcopy(source_trace)
            job = pool.apply_async(_single_failure_call, (trace,))

            try:
                generation, answer, evaluation = job.get(
                    timeout=TRACE_TIMEOUT_SECONDS
                )
            except mp.TimeoutError:
                timeout_trace = deepcopy(source_trace)
                timeout_trace["runtime"] = {
                    "timed_out": True,
                    "stage": f"single_failure_{failure_type}",
                    "timeout_seconds": TRACE_TIMEOUT_SECONDS,
                }
                timeouts.append(timeout_trace)
                print(
                    f"{failure_type} skipped trace {index}/{len(traces)}: "
                    f"TIMEOUT (> {TRACE_TIMEOUT_SECONDS:.0f}s)"
                )
                pool.terminate()
                pool.join()
                pool = ctx.Pool(processes=1)
                continue

            trace["experiment_stage"] = f"single_failure_result_{failure_type}"
            trace["failure_answer"] = answer
            trace["final_answer"] = answer
            trace["failure_evaluation"] = evaluation
            trace["evaluation"] = evaluation
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
            trace["prompt_hashes"] = prompt_hashes
            trace["latency_ms"] = int(generation["latency_ms"])
            trace["token_usage"] = generation["token_usage"]

            results.append(trace)
            print(f"{failure_type} processed trace {index}/{len(traces)}")

    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    timeout_path = _timeout_output_path(output_path)
    os.makedirs(os.path.dirname(timeout_path), exist_ok=True)
    with open(timeout_path, "w", encoding="utf-8") as file:
        json.dump(timeouts, file, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")
    print(f"Saved timeouts: {timeout_path} ({len(timeouts)})")


def main():
    for failure_type, input_path, output_path in EXPERIMENTS:
        run_single_failure(input_path, output_path, failure_type)


if __name__ == "__main__":
    mp.freeze_support()
    main()
