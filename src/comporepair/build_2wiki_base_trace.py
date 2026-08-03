import json
import uuid

from datasets import load_dataset
from dotenv import load_dotenv

from .pipeline.state import (
    CompoRepairTraceState,
    SelectedEvidence,
    VerificationState,
)

load_dotenv()


def main():

    print("1. Loading 2WikiMultihopQA dataset...")

    dataset = load_dataset(
        "framjay/2WikiMultihopQA",
        split="validation",
    )

    pilot_items = dataset.select(range(20))

    print("2. Normalizing data into the required trace schema...")

    traces = []

    for item in pilot_items:

        evidence_list = []

        supporting_facts = set(tuple(x) for x in item.get("supporting_facts", []))

        # 2Wiki context format:
        # [
        #   ["title", ["sentence1", "sentence2"]]
        # ]

        for index, context_item in enumerate(item["context"]):

            title = context_item[0]
            sentences = context_item[1]

            passage_text = " ".join(sentences)

            # Check whether this document contains
            # supporting sentences

            is_supporting = any(
                fact_title == title for fact_title, _ in supporting_facts
            )

            evidence = SelectedEvidence(
                passage_id=f"{item['_id']}_p_{index}",
                rank=index + 1,
                observable_signals={},
                title=title,
                text=passage_text,
                is_supporting=is_supporting,
                document_role=("supporting" if is_supporting else "non_supporting"),
            )

            evidence_list.append(evidence)

        trace = CompoRepairTraceState(
            trace_id=str(uuid.uuid4()),
            question_id=item["_id"],
            dataset="2wikimultihopqa",
            partition="pilot",
            experiment_stage="base_trace",
            dataset_metadata={
                "type": item.get("type", ""),
            },
            question=item["question"],
            canonical_answer=item["answer"],
            answer_aliases=[],
            retrieval_events=[],
            selected_evidence=evidence_list,
            answer_claims=[],
            verification=VerificationState(
                support=0.0, completeness=0.0, conflict=0.0, decision=""
            ),
            final_answer="",
            predicted_failures=[],
            true_failures=[],
            failure_after_repair=[],
            repair_history=[],
            model_manifest={},
            prompt_hashes={},
            latency_ms=0,
            token_usage={},
        )

        traces.append(trace)

    print("3. Saving processed traces...")

    with open(
        "data/processed/2wiki_base_traces.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            traces,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Done! Created data/processed/2wiki_base_traces.json")


if __name__ == "__main__":
    main()
