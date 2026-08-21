import json
import multiprocessing as mp
import os
from copy import deepcopy

from .pipeline.baseline_rag import model_manifest, run_baseline

INPUT_PATH = "data/processed/pilot_base_traces.json"
OUTPUT_PATH = "data/processed/pilot_baseline_results.json"
BASELINE_TRACE_TIMEOUT_SECONDS = 180.0


def _retrieval_events(documents, gold_supporting_passage_ids):
    gold_supporting_ids = {
        str(passage_id) for passage_id in gold_supporting_passage_ids
    }

    events = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        passage_id = str(metadata.get("passage_id", ""))

        is_supporting = passage_id in gold_supporting_ids

        events.append(
            {
                "title": metadata.get("title", ""),
                "passage_id": passage_id,
                "rank": index,
                "text": document.page_content,
                "score": metadata.get("score", 0.0),
                "is_supporting": is_supporting,
                "document_role": ("supporting" if is_supporting else "non_supporting"),
            }
        )

    return events


def _baseline_call(question, canonical_answer):
    return run_baseline(question, canonical_answer)


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        traces = json.load(file)

    results = []

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=1)

    try:
        for index, source_trace in enumerate(traces, start=1):
            trace = deepcopy(source_trace)

            job = pool.apply_async(
                _baseline_call,
                (
                    trace["question"],
                    trace["canonical_answer"],
                ),
            )

            try:
                result = job.get(timeout=BASELINE_TRACE_TIMEOUT_SECONDS)
            except mp.TimeoutError:
                print(
                    f"Baseline trace {index}/{len(traces)} timed out after "
                    f"{BASELINE_TRACE_TIMEOUT_SECONDS:.0f}s. Skipped."
                )
                pool.terminate()
                pool.join()
                pool = ctx.Pool(processes=1)
                continue

            answer = result.get("answer", "")
            evaluation = result["evaluation"]

            trace["experiment_stage"] = "baseline"
            trace["retrieval_events"] = _retrieval_events(
                result.get("retrieved_documents", []),
                trace.get("gold_supporting_passage_ids", []),
            )

            trace["baseline_answer"] = answer
            trace["failure_answer"] = ""
            trace["final_answer"] = answer

            trace["baseline_evaluation"] = evaluation
            trace["evaluation"] = evaluation

            trace["answer_claims"] = [
                {
                    "claim": answer,
                    "source": "baseline_generation",
                    "evidence_ids": [
                        item["passage_id"] for item in trace["retrieval_events"]
                    ],
                }
            ]

            trace["verification"] = {
                "support": 0.0,
                "completeness": 0.0,
                "conflict": 0.0,
                "decision": "not_evaluated",
            }

            trace["predicted_failures"] = []
            trace["true_failures"] = []
            trace["failure_history"] = []

            trace["failure_after_repair"] = []
            trace["new_failures_after_repair"] = []
            trace["regression_detected"] = False

            trace["repair_history"] = []

            trace["model_manifest"] = model_manifest()
            trace["prompt_hashes"] = {"answer_generation": result.get("prompt_hash", "")}

            trace["latency_ms"] = int(result.get("latency_ms", 0))
            trace["token_usage"] = result.get("token_usage", {})

            results.append(trace)
            print(f"Baseline processed trace {index}/{len(traces)}")

    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"Saved baseline results: {OUTPUT_PATH}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
