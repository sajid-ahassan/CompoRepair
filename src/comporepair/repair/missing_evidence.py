from copy import deepcopy
import re
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from ..pipeline.baseline_rag import (
    invoke_structured,
    llm,
)
from ..retrieval.vector_store import (
    REPAIR_CANDIDATE_K,
    RETRIEVAL_K,
    get_retriever,
)


M_REPAIR_VERSION = "m_original_reretrieval_conservative_compaction_v8"
M_REPAIR_STRATEGY = (
    "original_query_top_unseen_fill_then_conservative_sentence_compaction_v8"
)

COMPACTION_CONFIDENCE_THRESHOLD = 0.80
MIN_SENTENCES_TO_COMPACT = 3
MAX_SENTENCES_PER_DOCUMENT_FOR_PROMPT = 18


class DocumentCompactionDecision(BaseModel):
    document_label: str
    keep_full: bool
    relevant_sentence_ids: List[int] = []
    confidence: float = 0.0
    reason: str = ""


class CompactionPlan(BaseModel):
    decisions: List[DocumentCompactionDecision]


def _candidate_to_dict(document) -> Dict[str, Any]:
    metadata = document.metadata or {}
    return {
        "title": metadata.get("title", ""),
        "passage_id": metadata.get("passage_id", ""),
        "rank": 0,
        "text": document.page_content,
        "score": metadata.get("score", 0.0),
        "is_supporting": bool(metadata.get("is_supporting", False)),
        "document_role": metadata.get("document_role", "non_supporting"),
    }


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter with no external dependency.
    Falls back safely to the original full text if splitting is uncertain.
    """
    text = " ".join(str(text or "").split()).strip()
    if not text:
        return []

    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9"\'])',
        text,
    )
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not sentences:
        return [text]

    return sentences


def _format_text_only_documents(
    documents: List[Dict[str, Any]],
    protected_ids: set,
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Build neutral labels for the compaction LLM.

    Newly retrieved M-repair passages are protected and not sent for compaction.
    Passage titles, IDs, support labels, and ranks are hidden from the LLM.
    """
    blocks: List[str] = []
    label_map: Dict[str, Dict[str, Any]] = {}

    label_index = 1

    for document_index, document in enumerate(documents):
        passage_id = str(document.get("passage_id", "") or "")

        # Never compact newly recovered evidence.
        if passage_id and passage_id in protected_ids:
            continue

        sentences = _split_sentences(document.get("text", ""))

        if len(sentences) < MIN_SENTENCES_TO_COMPACT:
            continue

        # Avoid an unexpectedly huge structured-output prompt.
        prompt_sentences = sentences[:MAX_SENTENCES_PER_DOCUMENT_FOR_PROMPT]

        label = f"D{label_index}"
        label_index += 1

        label_map[label] = {
            "document_index": document_index,
            "sentences": sentences,
            "prompt_sentence_count": len(prompt_sentences),
        }

        sentence_lines = [
            f"S{sentence_index}: {sentence}"
            for sentence_index, sentence in enumerate(prompt_sentences, start=1)
        ]

        blocks.append(
            f"[{label}]\n" + "\n".join(sentence_lines)
        )

    return "\n\n".join(blocks), label_map


def _build_compaction_plan(
    question: str,
    documents: List[Dict[str, Any]],
    protected_ids: set,
) -> Tuple[Dict[str, DocumentCompactionDecision], str]:
    """
    One LLM call selects question-relevant sentences conservatively.

    It does not answer the question and does not use gold/reference data.
    """
    evidence_text, label_map = _format_text_only_documents(
        documents,
        protected_ids,
    )

    if not label_map:
        return {}, ""

    prompt = f"""Conservatively compact evidence for a multi-hop question.

Question:
{question}

Evidence documents:
{evidence_text}

For EACH document label:
- keep_full=true if the document may contain information needed for the answer,
  a bridge entity, identity alignment, comparison value, relation, date,
  number, profession, nationality, event, or any other reasoning step.
- keep_full=false ONLY when you are confident that some sentences are unrelated
  noise and the useful information can be preserved safely.
- relevant_sentence_ids must list every sentence that could help answer the
  question or connect one reasoning step to another.
- confidence is 0 to 1.

Important rules:
- Do NOT answer the question.
- Use only the shown sentence text.
- Do NOT infer from document order.
- Be conservative for multi-hop reasoning.
- Preserve bridge/identity sentences even when they do not contain the final
  answer directly.
- If uncertain, set keep_full=true.
- If keep_full=false, include ALL potentially useful sentence IDs.
- Never return an empty relevant_sentence_ids list when keep_full=false.
"""

    runnable = llm.with_structured_output(
        CompactionPlan,
        method="json_schema",
    )

    try:
        result = invoke_structured(runnable, prompt)
    except Exception as error:
        return {}, f"compaction_failed:{type(error).__name__}:{error}"

    decisions: Dict[str, DocumentCompactionDecision] = {}

    for decision in result.decisions:
        label = str(decision.document_label or "").strip()
        if label in label_map and label not in decisions:
            decisions[label] = decision

    return decisions, ""


def _apply_conservative_compaction(
    question: str,
    documents: List[Dict[str, Any]],
    protected_ids: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compact only pre-existing evidence, and only when the LLM is confident.

    Safety:
    - newly recovered passages remain byte-for-byte untouched
    - no document is removed
    - passage IDs/metadata remain unchanged
    - uncertain decisions keep the original full passage
    """
    compacted = [deepcopy(document) for document in documents]
    protected_id_set = {str(item) for item in protected_ids if item}

    evidence_text, label_map = _format_text_only_documents(
        compacted,
        protected_id_set,
    )

    if not label_map:
        return compacted, {
            "compaction_attempted": False,
            "compaction_applied": False,
            "documents_considered": 0,
            "documents_compacted_count": 0,
            "characters_before": sum(
                len(str(document.get("text", "") or ""))
                for document in compacted
            ),
            "characters_after": sum(
                len(str(document.get("text", "") or ""))
                for document in compacted
            ),
            "decisions": [],
            "technical_error": "",
        }

    decisions, technical_error = _build_compaction_plan(
        question,
        compacted,
        protected_id_set,
    )

    characters_before = sum(
        len(str(document.get("text", "") or ""))
        for document in compacted
    )

    audit_rows: List[Dict[str, Any]] = []
    compacted_count = 0

    if technical_error:
        return compacted, {
            "compaction_attempted": True,
            "compaction_applied": False,
            "documents_considered": len(label_map),
            "documents_compacted_count": 0,
            "characters_before": characters_before,
            "characters_after": characters_before,
            "decisions": [],
            "technical_error": technical_error,
        }

    for label, info in label_map.items():
        decision = decisions.get(label)
        document_index = int(info["document_index"])
        sentences = list(info["sentences"])
        prompt_sentence_count = int(info["prompt_sentence_count"])

        row = {
            "document_label": label,
            "passage_id": str(
                compacted[document_index].get("passage_id", "") or ""
            ),
            "keep_full": True,
            "confidence": 0.0,
            "selected_sentence_ids": [],
            "applied": False,
            "reason": "",
        }

        # Missing decision -> safest fallback is full passage.
        if decision is None:
            row["reason"] = "missing_decision_keep_full"
            audit_rows.append(row)
            continue

        row["keep_full"] = bool(decision.keep_full)
        row["confidence"] = float(decision.confidence)
        row["reason"] = str(decision.reason or "")

        if bool(decision.keep_full):
            audit_rows.append(row)
            continue

        if float(decision.confidence) < COMPACTION_CONFIDENCE_THRESHOLD:
            row["reason"] = (
                "low_confidence_keep_full:"
                + str(decision.reason or "")
            )
            audit_rows.append(row)
            continue

        valid_ids = sorted(
            {
                int(sentence_id)
                for sentence_id in decision.relevant_sentence_ids
                if isinstance(sentence_id, int)
                and 1 <= int(sentence_id) <= prompt_sentence_count
            }
        )

        if not valid_ids:
            row["reason"] = "empty_or_invalid_selection_keep_full"
            audit_rows.append(row)
            continue

        # If the passage had more sentences than were shown to the LLM,
        # preserve the unseen tail rather than deleting unseen information.
        selected_sentences = [
            sentences[sentence_id - 1]
            for sentence_id in valid_ids
        ]

        if len(sentences) > prompt_sentence_count:
            selected_sentences.extend(
                sentences[prompt_sentence_count:]
            )

        compacted_text = " ".join(selected_sentences).strip()
        original_text = str(
            compacted[document_index].get("text", "") or ""
        ).strip()

        if not compacted_text:
            row["reason"] = "empty_compacted_text_keep_full"
            audit_rows.append(row)
            continue

        if _normalise_text(compacted_text) == _normalise_text(original_text):
            row["selected_sentence_ids"] = valid_ids
            row["reason"] = "selection_equals_full_text_no_change"
            audit_rows.append(row)
            continue

        compacted[document_index]["text"] = compacted_text
        row["selected_sentence_ids"] = valid_ids
        row["applied"] = True
        compacted_count += 1
        audit_rows.append(row)

    characters_after = sum(
        len(str(document.get("text", "") or ""))
        for document in compacted
    )

    return compacted, {
        "compaction_attempted": True,
        "compaction_applied": compacted_count > 0,
        "documents_considered": len(label_map),
        "documents_compacted_count": compacted_count,
        "characters_before": characters_before,
        "characters_after": characters_after,
        "character_reduction": characters_before - characters_after,
        "decisions": audit_rows,
        "technical_error": "",
    }


# ---------------------------------------------------------------------------
# Original simple M retrieval.
#
# This helper is intentionally kept compatible with the prior codebase because
# the frozen D repair may import fill_to_context_size from this module.
# ---------------------------------------------------------------------------
def fill_to_context_size(
    question: str,
    documents: List[Dict[str, Any]],
    target_size: int = RETRIEVAL_K,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    selected = [deepcopy(document) for document in documents[:target_size]]

    selected_ids = {
        str(document.get("passage_id", ""))
        for document in selected
        if document.get("passage_id")
    }

    added_ids: List[str] = []

    if len(selected) < target_size:
        candidates = get_retriever(k=REPAIR_CANDIDATE_K).invoke(question)

        for candidate in candidates:
            candidate_dict = _candidate_to_dict(candidate)
            candidate_id = str(candidate_dict.get("passage_id", ""))

            if not candidate_id or candidate_id in selected_ids:
                continue

            selected.append(candidate_dict)
            selected_ids.add(candidate_id)
            added_ids.append(candidate_id)

            if len(selected) >= target_size:
                break

    for rank, document in enumerate(selected, start=1):
        document["rank"] = rank

    return selected, added_ids


def repair_missing_evidence(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    M v8.

    Flow:
        known M
        -> original question re-retrieval K=REPAIR_CANDIDATE_K
        -> restore highest-ranked unseen passage(s) toward RETRIEVAL_K
        -> protect newly recovered passage(s)
        -> conservatively compact only pre-existing evidence
        -> unchanged run_repair.py performs final answer generation

    No:
      - query generation
      - candidate reranker
      - post-repair answer gate
      - run_repair.py modification
      - gold/reference answer use
      - removed-passage ID use
      - failure-injection metadata use
    """
    repaired = deepcopy(trace)

    before_documents = [
        deepcopy(document)
        for document in repaired.get("retrieval_events", [])
    ]
    before_count = len(before_documents)

    # Step 1: use the original M repair that previously restored the injected
    # evidence reliably.
    restored_documents, added_ids = fill_to_context_size(
        repaired["question"],
        before_documents,
        target_size=RETRIEVAL_K,
    )

    # Step 2: compact only the OLD evidence. Newly recovered evidence is
    # protected from any text modification.
    compacted_documents, compaction_stats = _apply_conservative_compaction(
        repaired["question"],
        restored_documents,
        protected_ids=added_ids,
    )

    repaired["retrieval_events"] = compacted_documents

    history = list(repaired.get("repair_history", []))
    history.append(
        {
            "failure": "M",
            "repair": "expanded_reretrieval_plus_conservative_compaction",
            "repair_version": M_REPAIR_VERSION,
            "repair_strategy": M_REPAIR_STRATEGY,
            "context_count_before": before_count,
            "context_count_after": len(compacted_documents),
            "recovered_passage_ids": added_ids,
            "recovered_passage_count": len(added_ids),
            "context_restored": len(restored_documents) == RETRIEVAL_K,
            "retrieval_performed": before_count < RETRIEVAL_K,
            "retrieval_k": REPAIR_CANDIDATE_K,
            "original_query_used": True,
            "newly_retrieved_passages_protected_from_compaction": True,
            "compaction": compaction_stats,
            "gold_or_injection_metadata_used": False,
        }
    )

    repaired["repair_history"] = history
    return repaired