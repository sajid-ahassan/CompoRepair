import json
import uuid
from datasets import load_dataset
from .pipeline.state import CompoRepairTraceState, SelectedEvidence, VerificationState

from dotenv import load_dotenv

load_dotenv()


def main():
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    pilot_items = dataset.select(range(15))

    print("2. Normalizing data into the required trace schema...")
    traces = []
    for item in pilot_items:
        # Create a simple list to hold the evidence passages
        evidence_list = []
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        supporting_titles = item["supporting_facts"]["title"]

        # Loop through the raw context and build our evidence blocks
        for index, (title, sent_list) in enumerate(zip(titles, sentences)):
            passage_text = " ".join(sent_list)

            evidence = SelectedEvidence(
                passage_id=f"{item['id']}_p_{index}",
                rank=index + 1,
                observable_signals={},
                title=title,
                text=passage_text,
                is_supporting=title in supporting_titles,
                document_role=(
                    "supporting" if title in supporting_titles else "non_supporting"
                ),
            )
            evidence_list.append(evidence)

        # Build the final trace state dictionary
        trace = CompoRepairTraceState(
            trace_id=str(uuid.uuid4()),
            question_id=item["id"],
            dataset="hotpotqa",
            partition="pilot",
            experiment_stage="base_trace",
            dataset_metadata={"type": item["type"], "level": item["level"]},
            question=item["question"],
            canonical_answer=item["answer"],  # EVALUATION_ONLY[cite: 1]
            answer_aliases=[],
            retrieval_events=[],
            selected_evidence=evidence_list,
            answer_claims=[],
            verification=VerificationState(
                support=0.0, completeness=0.0, conflict=0.0, decision=""
            ),
            final_answer="",
            predicted_failures=[],
            true_failures=[],  # EVALUATION_ONLY[cite: 1]
            failure_after_repair=[],  # EVALUATION_ONLY[cite: 1]
            repair_history=[],
            model_manifest={},
            prompt_hashes={},
            latency_ms=0,
            token_usage={},
        )
        traces.append(trace)

    print("3. Saving processed traces to disk...")
    # Save the normalized data as a JSON file
    with open("data/processed/pilot_base_traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2)

    print("Done! Check 'data/processed/pilot_base_traces.json' to see your clean data.")


if __name__ == "__main__":
    main()
