import json
import re

from .pipeline.baseline_rag import run_baseline


def normalize_answer(answer):
    if answer is None:
        return ""
    answer = answer.lower()
    answer = re.sub(r"[^a-z0-9\s]", "", answer)
    answer = answer.strip()

    return answer


def evaluate_answer(prediction, ground_truth):

    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)

    return prediction == ground_truth


def main():

    with open("data/processed/pilot_base_traces.json", "r", encoding="utf-8") as f:
        traces = json.load(f)

    results = []
    i = 0
    for trace in traces:  # Limit to first 10 traces for testing

        result = run_baseline(trace["question"], trace["canonical_answer"])

        trace["final_answer"] = result["answer"]

        trace["evaluation"] = {
            "predicted_answer": normalize_answer(result["answer"]),
            "ground_truth": normalize_answer(trace["canonical_answer"]),
            "exact_match": evaluate_answer(result["answer"], trace["canonical_answer"]),
            "semantic_correct": result["evaluation"]["semantic_correct"],
        }

        trace["answer_claims"] = [
            {"claim": result["answer"], "source": "llm_generation"}
        ]

        trace["verification"] = {
            "support": 0.0,
            "completeness": 0.0,
            "conflict": 0.0,
            "decision": "not_evaluated",
        }

        trace["retrieval_events"] = [
            {
                "title": doc.metadata.get("title"),
                "passage_id": doc.metadata.get("passage_id"),
                "rank": index + 1,
                "text": doc.page_content,
            }
            for index, doc in enumerate(result["retrieved_documents"])
        ]

        results.append(trace)
        print(f"Baseline processed trace {i+1}")
        i += 1

    with open("data/processed/pilot_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(
        "Baseline evaluation completed. Results saved to 'data/processed/pilot_baseline_results.json'."
    )


if __name__ == "__main__":
    main()
