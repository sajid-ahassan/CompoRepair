from copy import deepcopy
from typing import Optional

from pydantic import BaseModel

from ..pipeline.baseline_rag import invoke_structured, llm
from ..retrieval.vector_store import RETRIEVAL_K
from .missing_evidence import fill_to_context_size

CONFIDENCE_THRESHOLD = 0.70


class DistractorRepairDecision(BaseModel):
    conflict_detected: bool
    remove_passage_id: Optional[str] = None
    confidence: float
    reason: str = ""


def detect_distractor(question, documents):
    id_map = {}
    passages = []
    for index, document in enumerate(documents, start=1):
        neutral_id = f"P{index}"
        id_map[neutral_id] = str(document.get("passage_id", ""))
        passages.append(
            f"{neutral_id} | Title: {document.get('title', '')}\n"
            f"{document.get('text', '')}"
        )

    evidence = "\n\n".join(passages)
    prompt = f"""Analyze all passages together and identify at most one passage that is clearly misleading for answering the question.

Question:
{question}

Passages:
{evidence}

Preserve bridge and entity-linking evidence. Never select both sides of a contradiction. If no single misleading passage can be identified confidently, set conflict_detected to false and remove_passage_id to null.
"""
    detector = llm.with_structured_output(
        DistractorRepairDecision,
        method="json_schema",
    )
    try:
        decision = invoke_structured(detector, prompt)
    except Exception as error:
        return {
            "conflict_detected": False,
            "remove_passage_id": None,
            "confidence": 0.0,
            "reason": f"detector_failed:{type(error).__name__}",
            "detector_failed": True,
        }

    return {
        "conflict_detected": decision.conflict_detected,
        "remove_passage_id": id_map.get(decision.remove_passage_id or ""),
        "confidence": decision.confidence,
        "reason": decision.reason,
        "detector_failed": False,
    }


def repair_distractor(trace):
    repaired = deepcopy(trace)
    documents = repaired.get("retrieval_events", [])
    decision = detect_distractor(repaired["question"], documents)

    remove_id = decision["remove_passage_id"]
    should_remove = (
        decision["conflict_detected"]
        and decision["confidence"] >= CONFIDENCE_THRESHOLD
        and remove_id is not None
    )
    filtered = [
        document
        for document in documents
        if not should_remove or str(document.get("passage_id", "")) != remove_id
    ]
    final_documents, replacement_ids = fill_to_context_size(
        repaired["question"],
        filtered,
        target_size=RETRIEVAL_K,
    )
    repaired["retrieval_events"] = final_documents
    repaired.setdefault("repair_history", []).append(
        {
            "failure": "D",
            "repair": "global_conflict_filtering",
            "decision": decision,
            "removed_passage_ids": [remove_id] if should_remove else [],
            "replacement_passage_ids": replacement_ids,
            "unresolved_conflict": decision["conflict_detected"] and not should_remove,
            "detector_failed": bool(decision.get("detector_failed", False)),
            "context_restored": len(final_documents) == RETRIEVAL_K,
        }
    )
    return repaired
