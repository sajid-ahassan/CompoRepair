from copy import deepcopy
from typing import Optional

from pydantic import BaseModel, model_validator

from ..pipeline.baseline_rag import invoke_structured, llm
from ..retrieval.vector_store import RETRIEVAL_K
from .missing_evidence import fill_to_context_size

CONFIDENCE_THRESHOLD = 0.70
D_REPAIR_VERSION = "d_retrieval_corroborated_pair_v3"


class DistractorRepairDecision(BaseModel):
    conflict_detected: bool
    remove_passage_id: Optional[str] = None
    conflicting_passage_id: Optional[str] = None
    confidence: float
    reason: str = ""

    @model_validator(mode="after")
    def validate_conflict_pair(self):
        """
        Enforce the detector contract.

        If a conflict is detected, the model must provide exactly two
        non-null, different passage IDs.

        Invalid structured output raises a Pydantic validation error.
        The existing invoke_structured() helper can then retry the call.
        """
        if self.conflict_detected:
            if not self.remove_passage_id:
                raise ValueError(
                    "remove_passage_id is required when conflict_detected=true"
                )

            if not self.conflicting_passage_id:
                raise ValueError(
                    "conflicting_passage_id is required when " "conflict_detected=true"
                )

            if self.remove_passage_id == self.conflicting_passage_id:
                raise ValueError(
                    "remove_passage_id and conflicting_passage_id " "must be different"
                )

        return self


def detect_distractor(question, documents):
    id_map = {}
    passages = []

    for index, document in enumerate(documents, start=1):
        neutral_id = f"P{index}"
        id_map[neutral_id] = str(document.get("passage_id", ""))
        passages.append(f"{neutral_id}\n{document.get('text', '')}")

    evidence = "\n\n".join(passages)

    prompt = f"""Analyze all passages together and identify at most one answer-relevant contradiction or misleading evidence conflict.

Question:
{question}

Passages:
{evidence}

Rules:

- Use only the passage text shown above. Do not use passage titles.
- Preserve bridge and entity-linking evidence.
- Focus on direct claim-level conflicts that matter for answering the question.
- Do not remove a passage merely because it is irrelevant, incomplete, or less detailed.
- When two passages make incompatible claims about the same answer-relevant entity, relation, number, date, place, role, or attribute, treat them as a conflict pair.
- If a conflict is found, return BOTH members of the pair:
  - remove_passage_id = the one you currently judge more likely to be misleading.
  - conflicting_passage_id = the other passage in the same conflict.
- remove_passage_id and conflicting_passage_id must be different P-IDs from the passages above.
- If conflict_detected=true, BOTH passage IDs are mandatory and must be valid P-IDs shown above.
- Never select both sides for removal.
- If no single answer-relevant conflict pair can be identified confidently, set conflict_detected=false and both passage IDs to null.
"""

    detector = llm.with_structured_output(
        DistractorRepairDecision,
        method="json_schema",
    )

    try:
        decision = invoke_structured(
            detector,
            prompt,
        )
    except Exception as error:
        return {
            "conflict_detected": False,
            "remove_passage_id": None,
            "conflicting_passage_id": None,
            "confidence": 0.0,
            "reason": f"detector_failed:{type(error).__name__}",
            "detector_failed": True,
        }

    return {
        "conflict_detected": bool(decision.conflict_detected),
        "remove_passage_id": id_map.get(decision.remove_passage_id or ""),
        "conflicting_passage_id": id_map.get(decision.conflicting_passage_id or ""),
        "confidence": float(decision.confidence),
        "reason": decision.reason,
        "detector_failed": False,
    }


def _remove_and_refill(
    question,
    documents,
    passage_id,
):
    filtered = [
        document
        for document in documents
        if str(document.get("passage_id", "")) != passage_id
    ]

    final_documents, replacement_ids = fill_to_context_size(
        question,
        filtered,
        target_size=RETRIEVAL_K,
    )

    reappeared = any(
        str(document.get("passage_id", "")) == passage_id
        for document in final_documents
    )

    return (
        final_documents,
        replacement_ids,
        reappeared,
    )


def repair_distractor(trace):
    repaired = deepcopy(trace)

    documents = repaired.get(
        "retrieval_events",
        [],
    )

    decision = detect_distractor(
        repaired["question"],
        documents,
    )

    current_ids = {
        str(document.get("passage_id", ""))
        for document in documents
        if document.get("passage_id")
    }

    primary_id = decision.get("remove_passage_id")

    partner_id = decision.get("conflicting_passage_id")

    eligible = (
        decision["conflict_detected"]
        and decision["confidence"] >= CONFIDENCE_THRESHOLD
        and not decision.get(
            "detector_failed",
            False,
        )
    )

    candidate_ids = []

    if eligible:
        for candidate_id in (
            primary_id,
            partner_id,
        ):
            if (
                candidate_id
                and candidate_id in current_ids
                and candidate_id not in candidate_ids
            ):
                candidate_ids.append(candidate_id)

    # D-v3 requires a complete pair.
    #
    # Even if the structured detector returned two
    # neutral P-IDs, one of them could theoretically
    # fail to map back to a current passage ID.
    #
    # In that case the repair must abstain.
    valid_pair = eligible and len(candidate_ids) == 2

    trials = {}

    if valid_pair:
        for candidate_id in candidate_ids:
            trials[candidate_id] = _remove_and_refill(
                repaired["question"],
                documents,
                candidate_id,
            )

    chosen_id = None
    selection_method = "no_confident_conflict"

    if valid_pair:
        first_id, second_id = candidate_ids

        first_reappeared = trials[first_id][2]

        second_reappeared = trials[second_id][2]

        # The frozen retriever acts as an
        # observable retrieval-persistence /
        # corroboration signal.
        #
        # If exactly one member of the conflict
        # pair immediately returns after removal,
        # preserve that side and remove the
        # opposite side.
        if first_reappeared and not second_reappeared:
            chosen_id = second_id
            selection_method = "retrieval_corroborated_pair"

        elif second_reappeared and not first_reappeared:
            chosen_id = first_id
            selection_method = "retrieval_corroborated_pair"

        else:
            # Both reappeared or neither
            # reappeared.
            #
            # The observable retrieval signal
            # is ambiguous, so abstain rather
            # than risk deleting useful evidence.
            selection_method = "ambiguous_pair_abstain"

    elif eligible:
        # Important D-v3 safety rule:
        #
        # A partial or invalid pair can never
        # cause evidence deletion.
        selection_method = "invalid_or_missing_pair_ids"

    if chosen_id is not None:
        (
            final_documents,
            replacement_ids,
            _,
        ) = trials[chosen_id]

        should_remove = True

    else:
        final_documents = documents
        replacement_ids = []
        should_remove = False

    repaired["retrieval_events"] = final_documents

    trial_log = {
        candidate_id: {
            "reappeared_after_refill": bool(trials[candidate_id][2]),
            "replacement_passage_ids": list(trials[candidate_id][1]),
        }
        for candidate_id in trials
    }

    repaired.setdefault(
        "repair_history",
        [],
    ).append(
        {
            "failure": "D",
            "repair": ("global_conflict_filtering"),
            "repair_version": (D_REPAIR_VERSION),
            "decision": decision,
            "selection_method": (selection_method),
            "retrieval_corroboration": (trial_log),
            "removed_passage_ids": ([chosen_id] if should_remove else []),
            "replacement_passage_ids": (replacement_ids),
            "unresolved_conflict": bool(
                decision["conflict_detected"] and not should_remove
            ),
            "detector_failed": bool(
                decision.get(
                    "detector_failed",
                    False,
                )
            ),
            "context_restored": (len(final_documents) == RETRIEVAL_K),
        }
    )

    return repaired
