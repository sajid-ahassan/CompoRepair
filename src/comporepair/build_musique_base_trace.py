import json
import os

from datasets import load_dataset
from dotenv import load_dotenv

from .pipeline.state import (
    CompoRepairTraceState,
    SelectedEvidence,
    VerificationState,
)

load_dotenv()

SAMPLE_START = int(os.getenv("COMPOREPAIR_SAMPLE_START", "0"))
SAMPLE_SIZE = int(os.getenv("COMPOREPAIR_SAMPLE_SIZE", "20"))
PARTITION = os.getenv("COMPOREPAIR_PARTITION", "pilot")
OUTPUT_PATH = "data/processed/pilot_base_traces.json"


def _question_id(item):
    return str(item.get("id"))


def _paragraphs(item):
    raw = item.get("paragraphs", [])
    if isinstance(raw, dict):
        return list(
            zip(
                raw.get("idx", []),
                raw.get("title", []),
                raw.get("paragraph_text", []),
                raw.get("is_supporting", []),
            )
        )
    return [
        (
            value["idx"],
            value["title"],
            value["paragraph_text"],
            value["is_supporting"],
        )
        for value in raw
    ]


def main():
    print("Loading MuSiQue dataset...")
    dataset = load_dataset("dgslibisey/MuSiQue", split="validation")

    if "answerable" in dataset.column_names:
        dataset = dataset.filter(lambda item: bool(item.get("answerable", False)))

    sample_end = min(SAMPLE_START + SAMPLE_SIZE, len(dataset))
    pilot_items = dataset.select(range(SAMPLE_START, sample_end))

    traces = []
    for pilot_index, item in enumerate(pilot_items):
        question_id = _question_id(item)

        evidence_list = []
        gold_supporting_passage_ids = []
        for index, (
            paragraph_idx,
            title,
            paragraph_text,
            is_supporting,
        ) in enumerate(_paragraphs(item)):
            title = str(title).strip()
            text = str(paragraph_text).strip()
            is_supporting = bool(is_supporting)
            passage_id = f"{question_id}_p_{paragraph_idx}"
            if is_supporting:
                gold_supporting_passage_ids.append(passage_id)

            evidence_list.append(
                SelectedEvidence(
                    passage_id=passage_id,
                    rank=index + 1,
                    observable_signals={
                        "paragraph_idx": int(paragraph_idx),
                        "context_index": index,
                        "supporting_sentences": [text] if is_supporting else [],
                    },
                    title=title,
                    text=text,
                    is_supporting=is_supporting,
                    document_role=("supporting" if is_supporting else "non_supporting"),
                )
            )

        traces.append(
            CompoRepairTraceState(
                trace_id=f"musique_{question_id}",
                question_id=question_id,
                dataset="musique",
                partition=PARTITION,
                experiment_stage="base_trace",
                dataset_metadata={
                    "split": "validation",
                    "sample_index": pilot_index,
                    "dataset_index": SAMPLE_START + pilot_index,
                    "answerable": bool(item.get("answerable", True)),
                },
                question=str(item["question"]).strip(),
                canonical_answer=str(item["answer"]).strip(),
                gold_supporting_passage_ids=gold_supporting_passage_ids,
                answer_aliases=[
                    str(alias).strip()
                    for alias in item.get("answer_aliases", [])
                    if str(alias).strip()
                ],
                retrieval_events=[],
                selected_evidence=evidence_list,
                answer_claims=[],
                verification=VerificationState(
                    support=0.0,
                    completeness=0.0,
                    conflict=0.0,
                    decision="",
                ),
                baseline_answer="",
                failure_answer="",
                final_answer="",
                predicted_failures=[],
                true_failures=[],
                failure_history=[],
                failure_after_repair=[],
                new_failures_after_repair=[],
                regression_detected=False,
                repair_history=[],
                model_manifest={},
                prompt_hashes={},
                latency_ms=0,
                token_usage={},
            )
        )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(traces, file, indent=2, ensure_ascii=False)

    print(f"Created {len(traces)} traces: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
