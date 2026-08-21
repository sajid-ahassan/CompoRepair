from copy import deepcopy
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Tuple

from pydantic import BaseModel, Field

from ..pipeline.baseline_rag import invoke_structured, llm
from ..retrieval.vector_store import get_retriever


L_REPAIR_VERSION = "l_same_source_refresh_plus_augmentation_v14"
L_RETRIEVAL_K = 20
L_SELECTED_K = 3
L_RERANK_BATCH_SIZE = 8
L_SAME_SOURCE_SIMILARITY_MIN = 0.90
L_SAME_SOURCE_LENGTH_RATIO_MIN = 0.85
L_SAME_SOURCE_LENGTH_RATIO_MAX = 1.15
L_CURRENT_SNIPPET_CHARS = 500
REPAIR_STRATEGY = "diagnose_retrieve_exact_dedup_same_source_refresh_rerank_append_v14"


class LinkRepairDiagnosis(BaseModel):
    repair_needed: bool = Field(
        description=(
            "True only when the current evidence plausibly contains an answer-relevant "
            "identity/reference/relation linkage problem that targeted retrieval could clarify."
        )
    )
    reason: str = Field(description="Short reason for the diagnosis.")
    broken_link_description: str = Field(
        description=(
            "Cautious description of the unclear answer-relevant connection. "
            "This is a retrieval-planning hypothesis, not factual evidence."
        )
    )
    retrieval_query: str = Field(
        description="One short targeted query for evidence that could clarify the broken link."
    )
    confidence: float = Field(ge=0.0, le=1.0)


class RetrievedPassageUtility(BaseModel):
    label: str = Field(description="Retrieved-evidence label, for example R3.")
    anchor_supported: bool = Field(
        description="True when the passage contains/refers to an entity or anchor needed by the question or broken link."
    )
    relation_supported: bool = Field(
        description="True when the passage states the missing or closely relevant relation/fact type."
    )
    bridge_candidate_present: bool = Field(
        description="True when the passage contains a plausible entity/value that could clarify the broken bridge."
    )
    reasoning_path_useful: bool = Field(
        description="True when the passage is useful for completing an answer-relevant reasoning hop."
    )
    irrelevant_or_wrong_relation: bool = Field(
        description="True when the passage is clearly irrelevant or mainly about the wrong relation."
    )
    reason: str = Field(description="Very short evidence-grounded reason.")


class RetrievedPassageUtilityBatch(BaseModel):
    assessments: List[RetrievedPassageUtility] = Field(default_factory=list)


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def _sanitize_retrieval_query(query: str, question: str) -> str:
    """Keep retrieval queries short and never search for the synthetic ENTITY_A token."""
    raw = _normalize_text(query)
    question_clean = _normalize_text(question)

    # ENTITY_A is an intervention artifact, not a real-world search anchor.
    if "entity_a" in raw.lower():
        raw = question_clean

    # Very long list-like queries hurt dense retrieval; fall back to the question.
    if len(raw.split()) > 22 or len(raw) > 220:
        raw = question_clean

    raw = raw.replace("ENTITY_A", " ").replace("entity_a", " ")
    raw = _normalize_text(raw)
    return raw or question_clean


def _text_similarity(a: Any, b: Any) -> float:
    left = _normalize_text(a).lower()
    right = _normalize_text(b).lower()
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def _length_ratio(a: Any, b: Any) -> float:
    left = len(_normalize_text(a))
    right = len(_normalize_text(b))
    if left <= 0 or right <= 0:
        return 0.0
    return min(left, right) / max(left, right)


def _documents(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        deepcopy(item)
        for item in trace.get("retrieval_events", [])
        if item.get("passage_id") and str(item.get("text", "")).strip()
    ]


def _format_current_passages(documents: Iterable[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"P{index}\nText: {item.get('text', '')}"
        for index, item in enumerate(documents, start=1)
    )


def _retrieved_to_dict(document: Any, rank: int) -> Dict[str, Any]:
    metadata = getattr(document, "metadata", None) or {}
    return {
        "title": str(metadata.get("title", "") or ""),
        "passage_id": str(metadata.get("passage_id", "") or ""),
        "rank": int(rank),
        "text": str(getattr(document, "page_content", "") or ""),
        "score": 0.0,
        # Repair retrieval never infers gold support status.
        "is_supporting": False,
        "document_role": "linkage_retrieved",
    }


def _retrieve_raw(query: str) -> List[Dict[str, Any]]:
    retriever = get_retriever(k=L_RETRIEVAL_K)
    raw = retriever.invoke(query)

    results: List[Dict[str, Any]] = []
    for rank, document in enumerate(raw, start=1):
        item = _retrieved_to_dict(document, rank)
        if item["passage_id"] and str(item["text"]).strip():
            results.append(item)
    return results


def _dedup_retrieved(
    current_documents: List[Dict[str, Any]],
    retrieved_documents: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deduplicate only by normalized text.

    Rules:
    - Empty retrievals are removed.
    - Exact-text copies of CURRENT evidence are removed.
    - Exact-text duplicates among retrieved results are removed.
    - Same passage_id with DIFFERENT text is allowed to remain.
    """
    current_texts = {
        _normalize_text(item.get("text", ""))
        for item in current_documents
        if _normalize_text(item.get("text", ""))
    }

    seen_retrieved_texts = set()
    candidates: List[Dict[str, Any]] = []
    removed_empty_ids: List[str] = []
    removed_exact_current_ids: List[str] = []
    removed_exact_retrieved_ids: List[str] = []

    current_ids = {
        str(item.get("passage_id", "")).strip()
        for item in current_documents
        if str(item.get("passage_id", "")).strip()
    }
    same_id_different_text_ids: List[str] = []

    for raw_rank, item in enumerate(retrieved_documents, start=1):
        passage_id = str(item.get("passage_id", "")).strip()
        normalized = _normalize_text(item.get("text", ""))

        if not passage_id or not normalized:
            if passage_id:
                removed_empty_ids.append(passage_id)
            continue

        if normalized in current_texts:
            removed_exact_current_ids.append(passage_id)
            continue

        if normalized in seen_retrieved_texts:
            removed_exact_retrieved_ids.append(passage_id)
            continue

        seen_retrieved_texts.add(normalized)
        candidate = deepcopy(item)
        candidate["_source_rank"] = int(raw_rank)
        candidate["_same_id_as_current"] = passage_id in current_ids
        if candidate["_same_id_as_current"] and passage_id not in same_id_different_text_ids:
            same_id_different_text_ids.append(passage_id)
        candidates.append(candidate)

    return candidates, {
        "removed_empty_count": len(removed_empty_ids),
        "removed_exact_current_duplicate_count": len(removed_exact_current_ids),
        "removed_exact_retrieved_duplicate_count": len(removed_exact_retrieved_ids),
        "removed_empty_ids": removed_empty_ids,
        "removed_exact_current_duplicate_ids": removed_exact_current_ids,
        "removed_exact_retrieved_duplicate_ids": removed_exact_retrieved_ids,
        "same_id_different_text_candidate_ids": same_id_different_text_ids,
        "same_id_different_text_candidate_count": len(same_id_different_text_ids),
    }


def _diagnose_link_problem(trace: Dict[str, Any]) -> LinkRepairDiagnosis:
    current_documents = _documents(trace)
    evidence = _format_current_passages(current_documents)
    current_answer = str(
        trace.get("failure_answer")
        or trace.get("final_answer")
        or ""
    )

    prompt = f"""
You are the DIAGNOSIS + TARGETED-RETRIEVAL stage of an Evidence-Linkage Repair system.

Question:
{trace.get('question', '')}

Current answer (diagnostic context only; it may be wrong):
{current_answer}

CURRENT evidence:
{evidence}

Evidence-Linkage Failure (L):
The current evidence may contain the needed facts, but an answer-relevant identity,
reference, entity alignment, or relation needed to connect the reasoning path is unclear.

Your task:
1. Decide whether targeted retrieval could help repair an answer-relevant linkage gap.
2. Describe the unclear connection cautiously.
3. Generate ONE short targeted retrieval query for evidence that could clarify the gap.

Rules:
- Do not invent the missing bridge.
- Do not choose a target passage.
- Do not rewrite evidence.
- Do not use passage titles, gold/support labels, canonical answers, failure_history,
  injected target IDs, or hidden metadata.
- Use only the question, current answer, and current passage text.
- If no linkage repair is needed, set repair_needed=false and return empty
  broken_link_description and retrieval_query.

The retrieval query should be short and focused on the entity/relation gap.
"""

    detector = llm.with_structured_output(
        LinkRepairDiagnosis,
        method="json_schema",
    )
    return invoke_structured(detector, prompt)


def _format_retrieved_chunk(
    labels: List[str],
    by_label: Dict[str, Dict[str, Any]],
) -> str:
    return "\n\n".join(
        f"{label}\nText: {by_label[label].get('text', '')}"
        for label in labels
    )


def _run_utility_batch(
    trace: Dict[str, Any],
    diagnosis: LinkRepairDiagnosis,
    current_documents: List[Dict[str, Any]],
    labels: List[str],
    by_label: Dict[str, Dict[str, Any]],
) -> List[RetrievedPassageUtility]:
    evidence = _format_retrieved_chunk(labels, by_label)
    current_context = "\n\n".join(
        f"P{i}\nText: {str(item.get('text', ''))[:L_CURRENT_SNIPPET_CHARS]}"
        for i, item in enumerate(current_documents, start=1)
    )

    prompt = f"""
You are the RETRIEVED-EVIDENCE UTILITY RERANKER for a multi-hop RAG repair system.

Question:
{trace.get('question', '')}

Broken-link hypothesis (search hint only; NOT factual evidence):
{diagnosis.broken_link_description}

CURRENT evidence (for context only):
{current_context}

NEW retrieved evidence candidates:
{evidence}

Assess EVERY shown R-label independently for whether adding it would help repair the
answer-relevant linkage that is still unclear in the CURRENT evidence.

For each label return:
- anchor_supported
- relation_supported
- bridge_candidate_present
- reasoning_path_useful
- irrelevant_or_wrong_relation
- a very short reason

Important:
- Score only the shown text and its complementarity with CURRENT evidence.
- Do not use titles, passage IDs, canonical answers, gold/support labels,
  failure metadata, source origin, or outside knowledge.
- Similar names alone are not proof of identity.
- Return one assessment for every shown R-label.
"""

    reranker = llm.with_structured_output(
        RetrievedPassageUtilityBatch,
        method="json_schema",
    )
    batch = invoke_structured(reranker, prompt)
    return list(batch.assessments or [])


def _score_retrieved_candidates(
    trace: Dict[str, Any],
    diagnosis: LinkRepairDiagnosis,
    current_documents: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Rerank RETRIEVED candidates only.

    Python utility score:
        +2 anchor_supported
        +3 relation_supported
        +2 bridge_candidate_present
        +3 reasoning_path_useful
        -4 irrelevant_or_wrong_relation

    The score orders candidates; selection also applies a logical usefulness guard.
    """
    if not candidates:
        return []

    by_label = {f"R{i}": item for i, item in enumerate(candidates, start=1)}
    labels = list(by_label)
    assessment_by_label: Dict[str, RetrievedPassageUtility] = {}

    for start in range(0, len(labels), L_RERANK_BATCH_SIZE):
        chunk = labels[start : start + L_RERANK_BATCH_SIZE]
        assessments = _run_utility_batch(trace, diagnosis, current_documents, chunk, by_label)

        mapped = 0
        for assessment in assessments:
            label = str(assessment.label or "").strip().upper()
            if label in chunk and label not in assessment_by_label:
                assessment_by_label[label] = assessment
                mapped += 1

        # Local 8B models sometimes return the right count with malformed labels.
        if mapped == 0 and len(assessments) == len(chunk):
            for label, assessment in zip(chunk, assessments):
                assessment_by_label[label] = assessment

    missing = [label for label in labels if label not in assessment_by_label]
    if missing:
        try:
            retry = _run_utility_batch(trace, diagnosis, current_documents, missing, by_label)
            for assessment in retry:
                label = str(assessment.label or "").strip().upper()
                if label in missing and label not in assessment_by_label:
                    assessment_by_label[label] = assessment
            still_missing = [label for label in missing if label not in assessment_by_label]
            if still_missing and len(retry) == len(missing):
                for label, assessment in zip(missing, retry):
                    if label not in assessment_by_label:
                        assessment_by_label[label] = assessment
        except Exception:
            pass

    scored: List[Dict[str, Any]] = []
    for label in labels:
        item = deepcopy(by_label[label])
        assessment = assessment_by_label.get(label)

        if assessment is None:
            score = 0
            features = {
                "anchor_supported": False,
                "relation_supported": False,
                "bridge_candidate_present": False,
                "reasoning_path_useful": False,
                "irrelevant_or_wrong_relation": False,
                "reason": "reranker_missing_assessment",
            }
        else:
            score = (
                2 * int(bool(assessment.anchor_supported))
                + 3 * int(bool(assessment.relation_supported))
                + 2 * int(bool(assessment.bridge_candidate_present))
                + 3 * int(bool(assessment.reasoning_path_useful))
                - 4 * int(bool(assessment.irrelevant_or_wrong_relation))
            )
            features = {
                "anchor_supported": bool(assessment.anchor_supported),
                "relation_supported": bool(assessment.relation_supported),
                "bridge_candidate_present": bool(assessment.bridge_candidate_present),
                "reasoning_path_useful": bool(assessment.reasoning_path_useful),
                "irrelevant_or_wrong_relation": bool(assessment.irrelevant_or_wrong_relation),
                "reason": str(assessment.reason or "").strip(),
            }

        item["_utility_label"] = label
        item["_utility_score"] = int(score)
        item["_utility_features"] = features
        scored.append(item)

    scored.sort(
        key=lambda item: (
            -int(item.get("_utility_score", 0)),
            int(item.get("_source_rank", 10**9)),
        )
    )
    return scored


def _is_useful_candidate(item: Dict[str, Any]) -> bool:
    """Logical usefulness guard; avoids appending clearly irrelevant top-k filler."""
    f = item.get("_utility_features", {}) or {}
    if bool(f.get("irrelevant_or_wrong_relation")):
        return False
    return bool(
        f.get("reasoning_path_useful")
        or f.get("bridge_candidate_present")
        or (f.get("anchor_supported") and f.get("relation_supported"))
    )


def _apply_high_confidence_same_source_refresh(
    current_documents: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deterministically refresh at most one stale current source.

    A retrieved passage is eligible only when:
    - its passage_id already exists in the current context,
    - its text is different,
    - text similarity is very high, and
    - lengths are close.

    This uses source identity only for source-refresh matching, never for utility scoring.
    All remaining candidates returned for augmentation are NEW passage IDs only.
    """
    refreshed = [deepcopy(item) for item in current_documents]
    current_by_id = {
        str(item.get("passage_id", "")).strip(): (idx, item)
        for idx, item in enumerate(refreshed)
        if str(item.get("passage_id", "")).strip()
    }

    assessed: List[Dict[str, Any]] = []
    eligible: List[Tuple[float, float, int, Dict[str, Any], int]] = []

    for item in candidates:
        pid = str(item.get("passage_id", "")).strip()
        if pid not in current_by_id:
            continue
        idx, current = current_by_id[pid]
        sim = _text_similarity(current.get("text", ""), item.get("text", ""))
        ratio = _length_ratio(current.get("text", ""), item.get("text", ""))
        ok = (
            sim >= L_SAME_SOURCE_SIMILARITY_MIN
            and L_SAME_SOURCE_LENGTH_RATIO_MIN <= ratio <= L_SAME_SOURCE_LENGTH_RATIO_MAX
        )
        assessed.append({
            "passage_id": pid,
            "source_rank": int(item.get("_source_rank", 0)),
            "text_similarity": round(sim, 6),
            "length_ratio": round(ratio, 6),
            "eligible": bool(ok),
        })
        if ok:
            eligible.append((sim, ratio, int(item.get("_source_rank", 10**9)), item, idx))

    refreshed_ids: List[str] = []
    if eligible:
        # One L intervention is injected per trace; refresh only the strongest matching source.
        eligible.sort(key=lambda x: (-x[0], -x[1], x[2]))
        sim, ratio, _, best, idx = eligible[0]
        pid = str(best.get("passage_id", "")).strip()
        refreshed[idx] = _clean_retrieved_for_trace(best, rank=idx + 1)
        refreshed[idx]["document_role"] = "linkage_source_refresh"
        refreshed_ids.append(pid)

    # Augmentation path is intentionally restricted to genuinely new source IDs.
    current_ids = set(current_by_id)
    new_id_candidates = [
        deepcopy(item)
        for item in candidates
        if str(item.get("passage_id", "")).strip() not in current_ids
    ]

    return refreshed, new_id_candidates, {
        "same_source_candidates_assessed": assessed,
        "same_source_candidate_count": len(assessed),
        "same_source_eligible_count": sum(1 for x in assessed if x["eligible"]),
        "source_refresh_applied": bool(refreshed_ids),
        "refreshed_passage_ids": refreshed_ids,
        "refreshed_count": len(refreshed_ids),
        "similarity_threshold": L_SAME_SOURCE_SIMILARITY_MIN,
        "length_ratio_min": L_SAME_SOURCE_LENGTH_RATIO_MIN,
        "length_ratio_max": L_SAME_SOURCE_LENGTH_RATIO_MAX,
    }


def _append_new_evidence(
    current_documents: List[Dict[str, Any]],
    selected_retrieved: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Append only genuinely new source IDs; never delete existing evidence."""
    merged = [deepcopy(item) for item in current_documents]
    existing_ids = {str(x.get("passage_id", "")).strip() for x in merged}
    existing_texts = {_normalize_text(x.get("text", "")) for x in merged}
    appended_ids: List[str] = []
    skipped_ids: List[str] = []

    for selected in selected_retrieved:
        pid = str(selected.get("passage_id", "")).strip()
        text = _normalize_text(selected.get("text", ""))
        if not pid or not text or pid in existing_ids or text in existing_texts:
            if pid:
                skipped_ids.append(pid)
            continue
        merged.append(_clean_retrieved_for_trace(selected, rank=len(merged) + 1))
        existing_ids.add(pid)
        existing_texts.add(text)
        appended_ids.append(pid)

    for rank, item in enumerate(merged, start=1):
        item["rank"] = rank

    return merged, {
        "appended_passage_ids": appended_ids,
        "appended_count": len(appended_ids),
        "skipped_passage_ids": skipped_ids,
        "skipped_count": len(skipped_ids),
    }


def _select_top_retrieved(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_ids = set()

    for item in scored:
        if not _is_useful_candidate(item):
            continue

        passage_id = str(item.get("passage_id", "")).strip()
        if not passage_id or passage_id in seen_ids:
            continue

        seen_ids.add(passage_id)
        selected.append(item)
        if len(selected) >= L_SELECTED_K:
            break

    return selected


def _clean_retrieved_for_trace(item: Dict[str, Any], rank: int) -> Dict[str, Any]:
    return {
        "title": str(item.get("title", "") or ""),
        "passage_id": str(item.get("passage_id", "") or ""),
        "rank": int(rank),
        "text": str(item.get("text", "") or ""),
        "score": float(item.get("score", 0.0) or 0.0),
        "is_supporting": False,
        "document_role": "linkage_retrieved",
    }


def _append_or_replace(
    current_documents: List[Dict[str, Any]],
    selected_retrieved: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Preserve the current context and merge selected retrievals conservatively.

    - Same passage_id as CURRENT + different text -> REPLACE that current instance.
    - New passage_id -> APPEND.
    - Exact normalized text already present -> SKIP.
    """
    merged = [deepcopy(item) for item in current_documents]
    id_to_index = {
        str(item.get("passage_id", "")).strip(): index
        for index, item in enumerate(merged)
        if str(item.get("passage_id", "")).strip()
    }
    current_texts = {
        _normalize_text(item.get("text", ""))
        for item in merged
        if _normalize_text(item.get("text", ""))
    }

    replaced_ids: List[str] = []
    appended_ids: List[str] = []
    exact_duplicate_skipped_ids: List[str] = []

    for selected in selected_retrieved:
        passage_id = str(selected.get("passage_id", "")).strip()
        normalized = _normalize_text(selected.get("text", ""))
        if not passage_id or not normalized:
            continue

        if normalized in current_texts:
            exact_duplicate_skipped_ids.append(passage_id)
            continue

        if passage_id in id_to_index:
            index = id_to_index[passage_id]
            old_normalized = _normalize_text(merged[index].get("text", ""))
            current_texts.discard(old_normalized)
            merged[index] = _clean_retrieved_for_trace(selected, rank=index + 1)
            current_texts.add(normalized)
            replaced_ids.append(passage_id)
        else:
            merged.append(_clean_retrieved_for_trace(selected, rank=len(merged) + 1))
            id_to_index[passage_id] = len(merged) - 1
            current_texts.add(normalized)
            appended_ids.append(passage_id)

    # Re-normalize ranks after all edits.
    for rank, item in enumerate(merged, start=1):
        item["rank"] = rank

    return merged, {
        "replaced_passage_ids": replaced_ids,
        "replaced_count": len(replaced_ids),
        "appended_passage_ids": appended_ids,
        "appended_count": len(appended_ids),
        "exact_duplicate_skipped_ids": exact_duplicate_skipped_ids,
        "exact_duplicate_skipped_count": len(exact_duplicate_skipped_ids),
    }


def _append_history(trace: Dict[str, Any], entry: Dict[str, Any]) -> None:
    trace.setdefault("repair_history", []).append(entry)


def _skip_result(
    trace: Dict[str, Any],
    *,
    reason: str,
    repair_needed: bool,
    technical_error: bool = False,
    detector_confidence: float = 0.0,
    broken_link_description: str = "",
    retrieval_query: str = "",
    retrieval_performed: bool = False,
    retrieved_count: int = 0,
    dedup_stats: Dict[str, Any] | None = None,
    candidate_count: int = 0,
) -> Dict[str, Any]:
    result = deepcopy(trace)
    _append_history(
        result,
        {
            "failure": "L",
            "repair": "same_source_refresh_plus_augmentation",
            "repair_version": L_REPAIR_VERSION,
            "repair_strategy": REPAIR_STRATEGY,
            "repair_needed": bool(repair_needed),
            "technical_error": bool(technical_error),
            "rewrite_applied": False,
            "link_patch_applied": False,
            "context_augmented": False,
            "source_refresh_applied": False,
            "evidence_refresh_applied": False,
            "reason": reason,
            "detector_confidence": float(detector_confidence),
            "broken_link_description": broken_link_description,
            "retrieval_query": retrieval_query,
            "retrieval_performed": bool(retrieval_performed),
            "retrieved_count": int(retrieved_count),
            "dedup_candidate_count": int(candidate_count),
            "dedup_stats": dict(dedup_stats or {}),
        },
    )
    return result


def repair_linkage(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    L repair v14: high-confidence same-source refresh + conservative augmentation.

    Flow:
        current context
        -> diagnose linkage gap + targeted query
        -> sanitize query (never search ENTITY_A; avoid giant list queries)
        -> retrieve K=20
        -> exact-text dedup
        -> if same source ID + highly similar different text: refresh ONE source directly
        -> rerank only genuinely NEW source IDs against the updated current evidence
        -> append up to 3 useful new passages
        -> preserve every untouched current passage
        -> outer repair runner regenerates the answer if evidence changed

    Leakage / safety:
    - No canonical answer, gold/support labels, failure_history, target ID, original clean text,
      or bridge identity is provided to the LLM repair stages.
    - Passage ID is used only by deterministic source-refresh matching after retrieval.
    - Passage ID is never shown to the utility reranker.
    - No synthetic bridge statement is generated.
    """
    original = deepcopy(trace)
    current_documents = _documents(original)

    if len(current_documents) < 2:
        return _skip_result(
            original,
            reason="fewer_than_two_current_passages",
            repair_needed=False,
        )

    # 1) Diagnose the linkage gap.
    try:
        diagnosis = _diagnose_link_problem(original)
    except Exception as error:
        return _skip_result(
            original,
            reason=f"L_detector_error:{type(error).__name__}:{error}",
            repair_needed=False,
            technical_error=True,
        )

    detector_confidence = float(diagnosis.confidence)
    broken_link_description = str(diagnosis.broken_link_description or "").strip()
    raw_query = str(diagnosis.retrieval_query or "").strip()

    if not diagnosis.repair_needed:
        return _skip_result(
            original,
            reason=str(diagnosis.reason or "").strip() or "no_linkage_repair_needed",
            repair_needed=False,
            detector_confidence=detector_confidence,
            broken_link_description=broken_link_description,
            retrieval_query=raw_query,
        )

    # 2) Sanitize the search query. Intervention artifacts must never become search anchors.
    retrieval_query = _sanitize_retrieval_query(
        raw_query or broken_link_description,
        str(original.get("question", "")),
    )
    if not retrieval_query:
        return _skip_result(
            original,
            reason="L_empty_retrieval_query",
            repair_needed=True,
            detector_confidence=detector_confidence,
            broken_link_description=broken_link_description,
        )

    # 3) Normal targeted retrieval.
    try:
        retrieved_documents = _retrieve_raw(retrieval_query)
    except Exception as error:
        return _skip_result(
            original,
            reason=f"L_retrieval_error:{type(error).__name__}:{error}",
            repair_needed=True,
            technical_error=True,
            detector_confidence=detector_confidence,
            broken_link_description=broken_link_description,
            retrieval_query=retrieval_query,
            retrieval_performed=True,
        )

    if not retrieved_documents:
        return _skip_result(
            original,
            reason="L_no_retrieved_evidence",
            repair_needed=True,
            detector_confidence=detector_confidence,
            broken_link_description=broken_link_description,
            retrieval_query=retrieval_query,
            retrieval_performed=True,
        )

    # 4) Exact-text dedup. Same-ID different-text candidates intentionally survive.
    candidates, dedup_stats = _dedup_retrieved(current_documents, retrieved_documents)
    if not candidates:
        return _skip_result(
            original,
            reason="L_no_new_retrieval_information",
            repair_needed=True,
            detector_confidence=detector_confidence,
            broken_link_description=broken_link_description,
            retrieval_query=retrieval_query,
            retrieval_performed=True,
            retrieved_count=len(retrieved_documents),
            dedup_stats=dedup_stats,
            candidate_count=0,
        )

    # 5) First path: deterministic high-confidence source refresh BEFORE reranking.
    refreshed_context, new_id_candidates, refresh_stats = _apply_high_confidence_same_source_refresh(
        current_documents, candidates
    )

    # 6) Second path: rerank only genuinely new sources, using updated current context.
    scored: List[Dict[str, Any]] = []
    selected: List[Dict[str, Any]] = []
    if new_id_candidates:
        try:
            scored = _score_retrieved_candidates(
                original, diagnosis, refreshed_context, new_id_candidates
            )
            selected = _select_top_retrieved(scored)
        except Exception as error:
            # A successful deterministic refresh should not be lost because augmentation failed.
            if not refresh_stats.get("source_refresh_applied"):
                return _skip_result(
                    original,
                    reason=f"L_reranker_error:{type(error).__name__}:{error}",
                    repair_needed=True,
                    technical_error=True,
                    detector_confidence=detector_confidence,
                    broken_link_description=broken_link_description,
                    retrieval_query=retrieval_query,
                    retrieval_performed=True,
                    retrieved_count=len(retrieved_documents),
                    dedup_stats=dedup_stats,
                    candidate_count=len(candidates),
                )

    # 7) Append at most 3 useful genuinely-new passages; never delete current evidence.
    merged, append_stats = _append_new_evidence(refreshed_context, selected)

    old_signature = [
        (str(item.get("passage_id", "")), _normalize_text(item.get("text", "")))
        for item in current_documents
    ]
    new_signature = [
        (str(item.get("passage_id", "")), _normalize_text(item.get("text", "")))
        for item in merged
    ]

    if old_signature == new_signature:
        reason = "L_no_useful_retrieved_evidence" if not selected else "L_merge_produced_no_evidence_change"
        return _skip_result(
            original,
            reason=reason,
            repair_needed=True,
            detector_confidence=detector_confidence,
            broken_link_description=broken_link_description,
            retrieval_query=retrieval_query,
            retrieval_performed=True,
            retrieved_count=len(retrieved_documents),
            dedup_stats={**dedup_stats, "refresh_stats": refresh_stats},
            candidate_count=len(candidates),
        )

    result = deepcopy(original)
    result["retrieval_events"] = merged

    selected_log = [
        {
            "passage_id": str(item.get("passage_id", "")),
            "source_rank": int(item.get("_source_rank", 0)),
            "utility_score": int(item.get("_utility_score", 0)),
            "utility_features": dict(item.get("_utility_features", {})),
        }
        for item in selected
    ]
    ranking_log = [
        {
            "passage_id": str(item.get("passage_id", "")),
            "source_rank": int(item.get("_source_rank", 0)),
            "utility_score": int(item.get("_utility_score", 0)),
            "utility_features": dict(item.get("_utility_features", {})),
        }
        for item in scored
    ]

    refreshed_count = int(refresh_stats.get("refreshed_count", 0))
    appended_count = int(append_stats.get("appended_count", 0))
    if refreshed_count and appended_count:
        applied_reason = "same_source_refresh_plus_new_evidence"
    elif refreshed_count:
        applied_reason = "high_confidence_same_source_refresh"
    else:
        applied_reason = "new_evidence_augmentation"

    _append_history(
        result,
        {
            "failure": "L",
            "repair": "same_source_refresh_plus_augmentation",
            "repair_version": L_REPAIR_VERSION,
            "repair_strategy": REPAIR_STRATEGY,
            "repair_needed": True,
            "technical_error": False,
            "rewrite_applied": False,
            "link_patch_applied": False,
            "context_augmented": bool(appended_count),
            "source_refresh_applied": bool(refreshed_count),
            "evidence_refresh_applied": True,
            "reason": applied_reason,
            "detector_reason": str(diagnosis.reason or "").strip(),
            "detector_confidence": detector_confidence,
            "broken_link_description": broken_link_description,
            "raw_retrieval_query": raw_query,
            "retrieval_query": retrieval_query,
            "retrieval_performed": True,
            "retrieval_k": L_RETRIEVAL_K,
            "retrieved_count": len(retrieved_documents),
            "current_context_count": len(current_documents),
            "dedup_candidate_count": len(candidates),
            "dedup_stats": dedup_stats,
            "refresh_stats": refresh_stats,
            "new_id_candidate_count": len(new_id_candidates),
            "selected_k_max": L_SELECTED_K,
            "selected_retrieved_count": len(selected),
            "selected_retrieved": selected_log,
            "utility_ranking": ranking_log,
            "append_stats": append_stats,
            "final_context_count": len(merged),
            "passage_id_used_in_utility_scoring": False,
            "passage_id_used_for_source_refresh_only": True,
            "same_source_refresh_similarity_guarded": True,
            "new_source_action": "append",
            "original_untouched_evidence_preserved": True,
            "synthetic_bridge_generated": False,
        },
    )

    return result
