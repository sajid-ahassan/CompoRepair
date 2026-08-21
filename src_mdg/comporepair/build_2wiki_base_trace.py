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
    return str(item.get("_id") or item.get("id"))


def _supporting_facts(item):
    raw = item.get("supporting_facts", [])
    if isinstance(raw, dict):
        return list(zip(raw.get("title", []), raw.get("sent_id", [])))
    return [tuple(value) for value in raw]


def _contexts(item):
    raw = item.get("context", [])
    if isinstance(raw, dict):
        return list(zip(raw.get("title", []), raw.get("sentences", [])))
    return [(value[0], value[1]) for value in raw]


def main():
    print("Loading 2WikiMultihopQA dataset...")
    dataset = load_dataset(
        "framolfese/2WikiMultihopQA",
        split="validation",
    )

    sample_end = min(SAMPLE_START + SAMPLE_SIZE, len(dataset))
    pilot_items = dataset.select(range(SAMPLE_START, sample_end))

    traces = []

    for pilot_index, item in enumerate(pilot_items):
        question_id = _question_id(item)

        supporting_map = {}
        for title, sent_id in _supporting_facts(item):
            supporting_map.setdefault(
                str(title).strip(),
                [],
            ).append(int(sent_id))

        evidence_list = []
        gold_supporting_passage_ids = []

        for index, (title, sentences) in enumerate(_contexts(item)):
            title = str(title).strip()

            text = " ".join(
                str(sentence).strip()
                for sentence in sentences
                if str(sentence).strip()
            ).strip()

            supporting_sentence_ids = supporting_map.get(title, [])
            is_supporting = bool(supporting_sentence_ids)
            passage_id = f"{question_id}_p_{index}"

            if is_supporting:
                gold_supporting_passage_ids.append(passage_id)

            evidence_list.append(
                SelectedEvidence(
                    passage_id=passage_id,
                    rank=index + 1,
                    observable_signals={
                        "supporting_sentence_ids": supporting_sentence_ids,
                        "context_index": index,
                    },
                    title=title,
                    text=text,
                    is_supporting=is_supporting,
                    document_role=(
                        "supporting"
                        if is_supporting
                        else "non_supporting"
                    ),
                )
            )

        traces.append(
            CompoRepairTraceState(
                trace_id=f"2wiki_{question_id}",
                question_id=question_id,
                dataset="2wikimultihopqa",
                partition=PARTITION,
                experiment_stage="base_trace",
                dataset_metadata={
                    "type": item.get("type", ""),
                    "split": "validation",
                    "sample_index": pilot_index,
                    "dataset_index": SAMPLE_START + pilot_index,
                },
                question=str(item["question"]).strip(),
                canonical_answer=str(item["answer"]).strip(),
                answer_aliases=[],
                gold_supporting_passage_ids=gold_supporting_passage_ids,
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
        json.dump(
            traces,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Created {len(traces)} traces: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()