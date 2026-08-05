from copy import deepcopy
from typing import Any, Dict, List, Tuple

from ..retrieval.vector_store import (
    REPAIR_CANDIDATE_K,
    RETRIEVAL_K,
    get_retriever,
)


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
    added_ids = []

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


def repair_missing_evidence(trace):
    repaired = deepcopy(trace)
    before_documents = repaired.get("retrieval_events", [])
    before_count = len(before_documents)

    final_documents, added_ids = fill_to_context_size(
        repaired["question"],
        before_documents,
        target_size=RETRIEVAL_K,
    )
    repaired["retrieval_events"] = final_documents

    history = list(repaired.get("repair_history", []))
    history.append(
        {
            "failure": "M",
            "repair": "expanded_reretrieval_fill",
            "context_count_before": before_count,
            "context_count_after": len(final_documents),
            "recovered_passage_ids": added_ids,
            "context_restored": len(final_documents) == RETRIEVAL_K,
        }
    )
    repaired["repair_history"] = history
    return repaired
