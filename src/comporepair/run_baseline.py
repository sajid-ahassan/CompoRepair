import json

from .pipeline.baseline_rag import run_baseline


def main():

    with open("data/processed/pilot_base_traces.json", "r", encoding="utf-8") as f:
        traces = json.load(f)

    results = []

    for trace in traces[:1]:

        result = run_baseline(trace["question"])

        trace["final_answer"] = result["answer"]
        
        trace["answer_claims"] = [
            {
                "claim": result["answer"],
                "source": "llm_generation"
            }
        ]

        trace["verification"] = {
            "support": 0.0,
            "completeness": 0.0,
            "conflict": 0.0,
            "decision": "not_evaluated"
        }

        trace["retrieval_events"] = [
            {
                "title": doc.metadata.get("title"),
                "passage_id": doc.metadata.get("passage_id"),
                'rank': doc.metadata.get("rank"),
            }
            for doc in result["retrieved_documents"]
        ]

        results.append(trace)

    with open("data/processed/pilot_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
