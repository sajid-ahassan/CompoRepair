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
SAMPLE_SIZE = int(os.getenv("COMPOREPAIR_SAMPLE_SIZE", "1200"))
SAMPLE_SEED = int(os.getenv("COMPOREPAIR_SAMPLE_SEED", "42"))
PARTITION = os.getenv("COMPOREPAIR_PARTITION", "pilot")

OUTPUT_PATH = "data/processed/pilot_base_traces.json"


ALLOWED_QUESTION_TYPES = {
    "compositional",
    "inference",
    "comparison",
    "bridge_comparison",
}


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

    return [(value[0], value[1]) for value in raw]


def _normalize_question_type(value: Any) -> str:
    """
    Normalize 2Wiki question type names.

    Examples:
        bridge-comparison -> bridge_comparison
        Bridge Comparison -> bridge_comparison
    """
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _select_items(
    dataset,
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Return a deterministic sample from all supported
    2WikiMultihopQA question types.

    Included:
        - compositional
        - inference
        - comparison
        - bridge_comparison

    No filtering based on the number of gold supporting passages.
    No artificial question-type ratio is imposed.
    """

    if SAMPLE_SIZE <= 0:
        raise ValueError("COMPOREPAIR_SAMPLE_SIZE must be greater than zero.")

    if SAMPLE_START < 0:
        raise ValueError("COMPOREPAIR_SAMPLE_START must be zero or greater.")

    eligible_items: List[Tuple[int, Dict[str, Any]]] = []

    for dataset_index in range(len(dataset)):
        item = dataset[dataset_index]

        question_type = _normalize_question_type(item.get("type", ""))

        if question_type not in ALLOWED_QUESTION_TYPES:
            continue

        eligible_items.append((dataset_index, item))

    # Deterministic shuffle.
    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(eligible_items)

    sample_end = SAMPLE_START + SAMPLE_SIZE

    selected = eligible_items[SAMPLE_START:sample_end]

    if len(selected) != SAMPLE_SIZE:
        raise ValueError(
            f"Requested {SAMPLE_SIZE} questions starting at "
            f"SAMPLE_START={SAMPLE_START}, but only "
            f"{len(selected)} questions were available."
        )

    return selected


def main() -> None:
    print("Loading 2WikiMultihopQA dataset...")

    dataset = load_dataset(
        "framolfese/2WikiMultihopQA",
        split="validation",
    )

    pilot_items = _select_items(dataset)

    # Count actual sampled question types.
    type_counts = {
        "compositional": 0,
        "inference": 0,
        "comparison": 0,
        "bridge_comparison": 0,
    }

    for _, item in pilot_items:
        question_type = _normalize_question_type(item.get("type", ""))

        if question_type in type_counts:
            type_counts[question_type] += 1

    print(
        f"Selected {len(pilot_items)} questions "
        f"using seed={SAMPLE_SEED}, "
        f"start={SAMPLE_START}."
    )

    print("Question-type distribution:")

    for question_type, count in type_counts.items():
        print(f"  {question_type}: {count}")

    traces = []

    for pilot_index, (
        dataset_index,
        item,
    ) in enumerate(pilot_items):

        question_id = _question_id(item)

        supporting_map: Dict[
            str,
            List[int],
        ] = {}

        for title, sent_id in _supporting_facts(item):

            supporting_map.setdefault(
                str(title).strip(),
                [],
            ).append(int(sent_id))

        evidence_list = []
        gold_supporting_passage_ids = []

        for index, (
            title,
            sentences,
        ) in enumerate(_contexts(item)):

            title = str(title).strip()

            text = " ".join(
                str(sentence).strip() for sentence in sentences if str(sentence).strip()
            ).strip()

            supporting_sentence_ids = supporting_map.get(
                title,
                [],
            )

            supporting_sentences = [
                str(sentences[sentence_id]).strip()
                for sentence_id in supporting_sentence_ids
                if (
                    0 <= sentence_id < len(sentences)
                    and str(sentences[sentence_id]).strip()
                )
            ]

            is_supporting = bool(supporting_sentence_ids)

            passage_id = f"{question_id}_p_{index}"

            if is_supporting:
                gold_supporting_passage_ids.append(passage_id)

            evidence_list.append(
                SelectedEvidence(
                    passage_id=passage_id,
                    rank=index + 1,
                    observable_signals={
                        "supporting_sentence_ids": (supporting_sentence_ids),
                        "supporting_sentences": (supporting_sentences),
                        "context_index": index,
                    },
                    title=title,
                    text=text,
                    is_supporting=is_supporting,
                    document_role=("supporting" if is_supporting else "non_supporting"),
                )
            )

        question_type = _normalize_question_type(item.get("type", ""))

        traces.append(
            CompoRepairTraceState(
                trace_id=f"2wiki_{question_id}",
                question_id=question_id,
                dataset="2wikimultihopqa",
                partition=PARTITION,
                experiment_stage="base_trace",
                dataset_metadata={
                    "type": question_type,
                    "split": "validation",
                    "sample_index": pilot_index,
                    "dataset_index": dataset_index,
                    "sampling_seed": SAMPLE_SEED,
                    "sample_start": SAMPLE_START,
                    "gold_supporting_passage_count": (len(gold_supporting_passage_ids)),
                },
                question=str(item["question"]).strip(),
                canonical_answer=str(item["answer"]).strip(),
                answer_aliases=[],
                gold_supporting_passage_ids=(gold_supporting_passage_ids),
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

    print(f"Created {len(traces)} traces: " f"{OUTPUT_PATH}")

    print("Final question-type distribution:")

    for question_type, count in type_counts.items():
        print(f"  {question_type}: {count}")

    # Additional useful audit:
    gold_count_distribution: Dict[int, int] = {}

    for trace in traces:
        count = len(
            trace.get(
                "gold_supporting_passage_ids",
                [],
            )
        )

        gold_count_distribution[count] = gold_count_distribution.get(count, 0) + 1

    print("Gold supporting-passage distribution:")

    for count in sorted(gold_count_distribution):
        print(f"  {count} gold passages: " f"{gold_count_distribution[count]}")


if __name__ == "__main__":
    main()
