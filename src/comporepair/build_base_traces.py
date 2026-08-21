import json
import os
import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from dotenv import load_dotenv

from .pipeline.state import (
    CompoRepairTraceState,
    SelectedEvidence,
    VerificationState,
)

load_dotenv()

SAMPLE_START = int(os.getenv("COMPOREPAIR_SAMPLE_START", "0"))
SAMPLE_SIZE = int(os.getenv("COMPOREPAIR_SAMPLE_SIZE", "2000"))
SAMPLE_SEED = int(os.getenv("COMPOREPAIR_SAMPLE_SEED", "42"))
PARTITION = os.getenv("COMPOREPAIR_PARTITION", "pilot")
OUTPUT_PATH = "data/processed/pilot_base_traces.json"


def _question_id(item: Dict[str, Any]) -> str:
    return str(item.get("_id") or item.get("id"))


def _supporting_facts(item: Dict[str, Any]) -> List[Tuple[Any, Any]]:
    raw = item.get("supporting_facts", [])

    if isinstance(raw, dict):
        return list(
            zip(
                raw.get("title", []),
                raw.get("sent_id", []),
            )
        )

    return [tuple(value) for value in raw]


def _contexts(item: Dict[str, Any]) -> List[Tuple[Any, Any]]:
    raw = item.get("context", [])

    if isinstance(raw, dict):
        return list(
            zip(
                raw.get("title", []),
                raw.get("sentences", []),
            )
        )

    return [
        (value[0], value[1])
        for value in raw
    ]


def _select_items(dataset) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Return a deterministic random sample from the full dataset.

    No filtering by bridge/comparison question type is performed.
    """

    if SAMPLE_SIZE <= 0:
        raise ValueError(
            "COMPOREPAIR_SAMPLE_SIZE must be greater than zero."
        )

    if SAMPLE_START < 0:
        raise ValueError(
            "COMPOREPAIR_SAMPLE_START must be zero or greater."
        )

    dataset_items = [
        (dataset_index, dataset[dataset_index])
        for dataset_index in range(len(dataset))
    ]

    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(dataset_items)

    selected = dataset_items[
        SAMPLE_START : SAMPLE_START + SAMPLE_SIZE
    ]

    if len(selected) != SAMPLE_SIZE:
        raise ValueError(
            f"Requested {SAMPLE_SIZE} questions starting from "
            f"{SAMPLE_START}, but only {len(selected)} are available."
        )

    return selected


def main() -> None:
    print("Loading HotpotQA dataset...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )

    pilot_items = _select_items(dataset)

    print(
        f"Selected {len(pilot_items)} questions "
        f"from the full validation dataset."
    )

    traces = []

    for pilot_index, (dataset_index, item) in enumerate(pilot_items):
        question_id = _question_id(item)

        supporting_map: Dict[str, List[int]] = {}

        for title, sent_id in _supporting_facts(item):
            supporting_map.setdefault(
                str(title).strip(),
                [],
            ).append(int(sent_id))

        evidence_list = []
        gold_supporting_passage_ids = []

        for index, (title, sentences) in enumerate(
            _contexts(item)
        ):
            title = str(title).strip()

            text = " ".join(
                str(sentence).strip()
                for sentence in sentences
                if str(sentence).strip()
            ).strip()

            supporting_sentence_ids = supporting_map.get(
                title,
                [],
            )

            supporting_sentences = [
                str(sentences[sentence_id]).strip()
                for sentence_id in supporting_sentence_ids
                if 0 <= sentence_id < len(sentences)
                and str(sentences[sentence_id]).strip()
            ]

            is_supporting = bool(
                supporting_sentence_ids
            )

            passage_id = (
                f"{question_id}_p_{index}"
            )

            if is_supporting:
                gold_supporting_passage_ids.append(
                    passage_id
                )

            evidence_list.append(
                SelectedEvidence(
                    passage_id=passage_id,
                    rank=index + 1,
                    observable_signals={
                        "supporting_sentence_ids":
                            supporting_sentence_ids,
                        "supporting_sentences":
                            supporting_sentences,
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
                trace_id=f"hotpot_{question_id}",
                question_id=question_id,
                dataset="hotpotqa",
                partition=PARTITION,
                experiment_stage="base_trace",
                dataset_metadata={
                    "type": str(
                        item.get("type", "")
                    ).strip().lower(),
                    "level": item.get(
                        "level",
                        "",
                    ),
                    "split": "validation",
                    "sample_index": pilot_index,
                    "dataset_index": dataset_index,
                    "sampling_seed": SAMPLE_SEED,
                },
                question=str(
                    item["question"]
                ).strip(),
                canonical_answer=str(
                    item["answer"]
                ).strip(),
                answer_aliases=[],
                gold_supporting_passage_ids=(
                    gold_supporting_passage_ids
                ),
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

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            traces,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Created {len(traces)} traces: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()