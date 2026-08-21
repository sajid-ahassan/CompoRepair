import json
import multiprocessing as mp
import os
import time
import unicodedata
from copy import deepcopy
from typing import Any, Dict, List, Literal, Tuple

from pydantic import BaseModel

from .pipeline.baseline_rag import (
    evaluate_prediction,
    format_evidence,
    generate_answer,
    invoke_structured,
    llm,
    model_manifest,
)
from .run_repair import apply_repair, diagnose_failures_after_repair

SAFE_COMPOSER_VERSION = "safe_composer_v5_grounded_structural_safety_gate"
SAFE_PLANNER_VERSION = "neutral_trace_specific_order_v2_frozen"
SAFE_GATE_VERSION = "grounded_question_conditioned_structural_gate_v5"
SAFE_V5_OUTPUT_ROOT = (
    os.getenv(
        "COMPOREPAIR_SAFE_V5_OUTPUT_ROOT",
        "data/repaired_result",
    ).strip()
    or "data/repaired_result"
)

EXPERIMENTS = [
    # (
    #     "M_D_safe",
    #     "data/results/compound/M_D_results.json",
    #     os.path.join(SAFE_V5_OUTPUT_ROOT, "M_D_safe_repair_results.json"),
    # ),
    # (
    #     "M_L_safe",
    #     "data/results/compound/M_L_results.json",
    #     os.path.join(SAFE_V5_OUTPUT_ROOT, "M_L_safe_repair_results.json"),
    # ),
    # (
    #     "D_L_safe",
    #     "data/results/compound/D_L_results.json",
    #     os.path.join(SAFE_V5_OUTPUT_ROOT, "D_L_safe_repair_results.json"),
    # ),
    (
        "M_D_L_safe",
        "data/results/compound/M_D_L_results.json",
        os.path.join(SAFE_V5_OUTPUT_ROOT, "M_D_L_safe_repair_results.json"),
    ),
]

MAX_SAFE_STEPS = 3
MAX_TRACE_SECONDS = 120.0


def _log(message: str) -> None:
    """Print progress immediately so long model calls are visible."""
    print(message, flush=True)


def _evidence_signature(trace: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Match run_repair.py's evidence-change definition exactly.

    Passage order, passage id, and passage text define the observable evidence
    state. Repair-history/metadata-only changes are intentionally ignored.
    """
    return tuple(
        (
            str(item.get("passage_id", "")),
            str(item.get("text", "")),
        )
        for item in trace.get("retrieval_events", [])
    )


def _atomic_save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Checkpoint safely so an interruption does not corrupt completed work."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    temp_path = f"{output_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    os.replace(temp_path, output_path)


def _load_checkpoint(output_path: str) -> List[Dict[str, Any]]:
    """Load completed traces from a checkpoint safely.

    Empty/corrupted checkpoints can occur after an interrupted manual setup or
    from older non-atomic runs. Safe-v5 starts fresh for an empty file, but it
    does NOT silently discard a non-empty malformed checkpoint: that case is
    raised so completed work is not accidentally overwritten.
    """
    if not os.path.exists(output_path):
        return []

    if os.path.getsize(output_path) == 0:
        _log(f"Checkpoint is empty: {output_path}. Starting fresh.")
        return []

    try:
        with open(output_path, "r", encoding="utf-8") as file:
            value = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Checkpoint JSON is malformed: {output_path}. "
            f"Refusing to overwrite it automatically. Original error: {error}"
        ) from error

    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in checkpoint: {output_path}")
    return value


def _force_restart_requested() -> bool:
    return os.getenv("COMPOREPAIR_SAFE_FORCE_RESTART", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


class SafeRepairOrderDecision(BaseModel):
    repair_order: List[Literal["M", "D", "L"]]
    reason: str


class EvidenceSupportRef(BaseModel):
    passage_id: str
    quote: str


class SafeGroundingAssessment(BaseModel):
    before_entailment: Literal["entailed", "partial", "unsupported", "contradicted"]
    after_entailment: Literal["entailed", "partial", "unsupported", "contradicted"]
    before_answers_question: bool
    after_answers_question: bool
    after_has_unsupported_claim: bool
    new_conflict_introduced: bool
    useful_evidence_lost: bool
    before_support: List[EvidenceSupportRef]
    after_support: List[EvidenceSupportRef]
    reason: str


def _available_repairs(trace: Dict[str, Any]) -> List[str]:
    """Controlled experiment routing: expose only injected compound modules.

    Fixed/reverse baselines use the same true-failure labels to decide which
    primitive modules are available. The safe composer changes only the order
    and whether each attempted step is kept or reverted.
    """
    failures = list(trace.get("true_failures", []))
    return [failure for failure in ("M", "D", "L") if failure in failures]


def _normalise_order(order: List[str], failures: List[str]) -> List[str]:
    """Return each available repair exactly once, with deterministic fallback."""
    allowed = [failure for failure in ("M", "D", "L") if failure in failures]
    result: List[str] = []
    for failure in order:
        if failure in allowed and failure not in result:
            result.append(failure)
    for failure in allowed:
        if failure not in result:
            result.append(failure)
    return result[:MAX_SAFE_STEPS]


def _validate_planner_order(order: List[str], failures: List[str]) -> List[str]:
    """Require the planner itself to return one exact permutation.

    V1 silently completed incomplete planner outputs in M/D/L order, which could
    create an artificial default-order bias. V2 treats an incomplete, repeated,
    or invalid plan as a planner technical failure and uses the explicit fallback
    path instead.
    """
    allowed = [failure for failure in ("M", "D", "L") if failure in failures]
    raw = [str(item) for item in order]
    if (
        len(raw) != len(allowed)
        or len(set(raw)) != len(raw)
        or set(raw) != set(allowed)
    ):
        raise ValueError(
            f"invalid_planner_order: expected permutation of {allowed}, got {raw}"
        )
    return raw[:MAX_SAFE_STEPS]


def _observed_answer(trace: Dict[str, Any]) -> str:
    return str(
        trace.get("failure_answer")
        or trace.get("final_answer")
        or trace.get("baseline_answer")
        or ""
    )


def _choose_repair_order(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Choose one trace-specific repair order with a neutral planner.

    V2 intentionally removes directional hints such as "M before D before L".
    The planner must treat the available labels as an unordered set and justify
    its order from the observable state of this trace only.
    """
    allowed = _available_repairs(trace)
    current_answer = _observed_answer(trace)

    # Definitions describe what each primitive does; their presentation order is
    # explicitly declared non-prioritized in the prompt.
    module_definitions = {
        "M": "retrieve and add potentially missing answer-relevant evidence",
        "D": "detect and remove one clearly misleading passage, then refill context",
        "L": "restore an unclear cross-passage link through retrieval-supported rewriting",
    }
    allowed_text = ", ".join(allowed)
    definitions_text = "\n".join(
        f"{label} = {module_definitions[label]}"
        for label in ("D", "L", "M")
        if label in allowed
    )

    prompt = f"""Choose the safest ONE-TIME repair order for THIS trace.

Question:
{trace['question']}

Current evidence:
{format_evidence(trace.get('retrieval_events', []))}

Current generated answer:
{current_answer}

Available repair labels are the UNORDERED set: {{{allowed_text}}}

Module definitions (the display order below is arbitrary and is NOT a priority):
{definitions_text}

Planning rules:
- Decide the order independently for this trace from the observable question, evidence, and current answer.
- Compare the possible orders before choosing.
- Identify which repair most directly addresses the largest observable obstacle first.
- Consider whether applying one repair first could help or interfere with a later repair.
- Do NOT assume M, D, or L normally comes first. Use the best order that you can justify for this trace.
- Do NOT copy the order in which labels or definitions are displayed.
- Do NOT choose a different order merely for diversity. If one order is genuinely best for this trace, choose it.
- Return every available repair label exactly once.
- Never repeat a repair label.
- Maximum length is three.
- Preserve useful evidence whenever possible.
- This is a one-time plan: there will be no replanning and no repair retries.
- Do not use passage titles as evidence.
- Do not assume access to a reference answer.

In `reason`, briefly cite the observable trace-specific reason for the chosen first step and the main interaction consideration for the remaining order.
"""

    planner = llm.with_structured_output(
        SafeRepairOrderDecision,
        method="json_schema",
    )
    try:
        decision = invoke_structured(planner, prompt)
        raw_order = list(decision.repair_order)
        order = _validate_planner_order(raw_order, allowed)
        return {
            "repair_order": order,
            "raw_repair_order": raw_order,
            "reason": str(decision.reason or ""),
            "technical_error": "",
            "planner_version": SAFE_PLANNER_VERSION,
        }
    except Exception as error:
        fallback = _normalise_order([], allowed)
        return {
            "repair_order": fallback,
            "raw_repair_order": [],
            "reason": "Deterministic fallback order used after planner failure.",
            "technical_error": f"planner_failed:{type(error).__name__}:{error}",
            "planner_version": SAFE_PLANNER_VERSION,
        }


SUPPORT_SCORE = {
    "contradicted": 0,
    "unsupported": 1,
    "partial": 2,
    "entailed": 3,
}


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )
    return " ".join(text.split())


def _format_grounding_evidence(events: List[Dict[str, Any]]) -> str:
    """Format observable evidence without titles and with persistent passage IDs."""
    blocks: List[str] = []
    for index, item in enumerate(events, start=1):
        passage_id = str(item.get("passage_id", ""))
        text = str(item.get("text", ""))
        blocks.append(f"[P{index}] PASSAGE_ID={passage_id}\n{text}")
    return "\n\n".join(blocks)


def _validate_support_refs(
    refs: List[EvidenceSupportRef],
    trace: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Validate that each cited quote really occurs in the claimed passage.

    The LLM may assess entailment, but it cannot invent evidence. Passage IDs and
    extractive quotes are checked deterministically against the observable state.
    """
    by_id = {
        str(item.get("passage_id", "")): str(item.get("text", ""))
        for item in trace.get("retrieval_events", [])
        if str(item.get("passage_id", ""))
    }
    valid: List[Dict[str, str]] = []
    invalid: List[Dict[str, str]] = []
    seen = set()

    for ref in refs or []:
        passage_id = str(ref.passage_id or "")
        quote = str(ref.quote or "").strip()
        key = (passage_id, quote)
        if key in seen:
            continue
        seen.add(key)

        passage_text = by_id.get(passage_id, "")
        quote_norm = _normalise_text(quote)
        passage_norm = _normalise_text(passage_text)

        # Reject empty/trivial citations. Two words are enough for short factual
        # answers such as "Conservative Party", while still preventing a single
        # generic token from being treated as grounding.
        quote_token_count = len(quote_norm.split())
        quote_ok = len(quote_norm) >= 8 and quote_token_count >= 2
        matched = bool(quote_ok and quote_norm in passage_norm)

        record = {"passage_id": passage_id, "quote": quote}
        if matched:
            valid.append(record)
        else:
            invalid.append(record)

    return valid, invalid


def _grounded_entailment(
    raw_label: str,
    answers_question: bool,
    valid_support: List[Dict[str, str]],
) -> str:
    """Downgrade unsupported LLM entailment claims deterministically."""
    label = str(raw_label or "unsupported")
    if label not in SUPPORT_SCORE:
        label = "unsupported"

    if label in {"entailed", "partial"} and not valid_support:
        return "unsupported"

    # A response that does not actually answer the question cannot count as
    # positively entailed, even if it mentions a relevant entity somewhere.
    if not answers_question and label in {"entailed", "partial"}:
        return "unsupported"

    return label


def _looks_like_non_answer(answer: str) -> bool:
    """Catch common refusal/uncertainty outputs before they are accepted."""
    prefix = _normalise_text(answer)[:260]
    patterns = (
        "i cannot answer",
        "i can't answer",
        "i cant answer",
        "i cannot identify",
        "i couldn't find",
        "i couldnt find",
        "unable to determine",
        "not enough information",
        "not specified in the provided evidence",
        "not specified in provided evidence",
        "not explicitly stated in the provided evidence",
        "not explicitly stated in provided evidence",
    )
    return any(pattern in prefix for pattern in patterns)


def _structural_progress(
    repair_label: str,
    attempted_history: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Use only the primitive's own observable repair outputs, never gold labels."""
    entries = [
        item
        for item in attempted_history
        if str(item.get("failure", "")) == repair_label
    ]

    if repair_label == "M":
        for item in entries:
            recovered = list(item.get("recovered_passage_ids", []) or [])
            before_count = item.get("context_count_before")
            after_count = item.get("context_count_after")
            count_increased = (
                isinstance(before_count, int)
                and isinstance(after_count, int)
                and after_count > before_count
            )
            if recovered or count_increased:
                return (
                    True,
                    f"M recovered={len(recovered)} context_increased={count_increased}",
                )
        return False, "M produced no observable retrieval restoration"

    if repair_label == "D":
        for item in entries:
            removed = list(item.get("removed_passage_ids", []) or [])
            detector_failed = bool(item.get("detector_failed", False))
            unresolved = bool(item.get("unresolved_conflict", False))
            if removed and not detector_failed:
                return (
                    True,
                    f"D removed={len(removed)} unresolved_conflict={unresolved}",
                )
        return False, "D removed no passage"

    if repair_label == "L":
        for item in entries:
            rewrite_applied = bool(item.get("rewrite_applied", False))
            technical_error = bool(item.get("technical_error", False))
            same_id_count = int(item.get("primary_same_id_candidate_count", 0) or 0)
            if rewrite_applied and not technical_error:
                return (
                    True,
                    f"L rewrite_applied=true primary_same_id_count={same_id_count}",
                )
        return False, "L produced no successful rewrite"

    return False, "unknown repair label"


def _answer_changed(before_answer: str, after_answer: str) -> bool:
    return _normalise_text(before_answer) != _normalise_text(after_answer)


def _decide_v5_gate(
    *,
    answer_changed: bool,
    before_entailment: str,
    after_entailment: str,
    before_score: int,
    after_score: int,
    after_answers: bool,
    non_answer: bool,
    unsupported_claim: bool,
    new_conflict: bool,
    useful_loss: bool,
    valid_after: List[Dict[str, str]],
    structural_progress: bool,
) -> Dict[str, Any]:
    """Pure deterministic V5 KEEP/REVERT policy.

    This helper intentionally contains no LLM calls and no gold/reference-answer
    access. It separates observable harm from observable progress so the policy
    can be unit-tested independently of model inference.
    """
    hard_harm_reasons: List[str] = []
    support_decreased = after_score < before_score
    severe_support_drop = (
        after_score < SUPPORT_SCORE["partial"]
        and before_score >= SUPPORT_SCORE["partial"]
    )
    newly_contradicted = (
        after_entailment == "contradicted" and before_entailment != "contradicted"
    )

    if new_conflict:
        hard_harm_reasons.append("new_conflict_introduced")

    if newly_contradicted:
        hard_harm_reasons.append("newly_contradicted")

    if answer_changed:
        # New answer claim: require strict, validated question-conditioned support.
        if not after_answers:
            hard_harm_reasons.append("changed_answer_does_not_answer_question")
        if non_answer:
            hard_harm_reasons.append("changed_answer_is_refusal_or_non_answer")
        if unsupported_claim:
            hard_harm_reasons.append("changed_answer_has_unsupported_claim")
        if useful_loss:
            hard_harm_reasons.append("useful_evidence_lost")
        if after_entailment != "entailed":
            hard_harm_reasons.append("changed_answer_not_fully_entailed")
        if not valid_after:
            hard_harm_reasons.append("changed_answer_has_no_valid_support_quote")
        if support_decreased:
            hard_harm_reasons.append("question_conditioned_support_decreased")

        improvement = after_score > before_score
        accepted_progress = structural_progress or improvement
        decision_policy = "changed_answer_strict_grounding"

    else:
        # Same answer: allow verified intermediate structural progress. A compound
        # repair often needs M/D to improve the evidence state before a later step
        # can make the answer fully grounded. Do not reject that progress merely
        # because the current answer is still unsupported.
        if severe_support_drop:
            hard_harm_reasons.append("severe_question_conditioned_support_drop")
        if useful_loss and (support_decreased or not structural_progress):
            hard_harm_reasons.append("useful_evidence_lost_with_no_safe_offset")

        improvement = after_score > before_score
        accepted_progress = structural_progress or improvement
        decision_policy = "unchanged_answer_structural_progress"

    keep_change = bool(not hard_harm_reasons and accepted_progress)
    if not hard_harm_reasons and not accepted_progress:
        hard_harm_reasons.append("no_grounded_or_structural_progress")
        keep_change = False

    return {
        "keep_change": keep_change,
        "hard_harm_reasons": hard_harm_reasons,
        "decision_policy": decision_policy,
        "support_improved": improvement,
        "accepted_progress": accepted_progress,
    }


def _judge_step(
    question: str,
    repair_label: str,
    before_trace: Dict[str, Any],
    after_trace: Dict[str, Any],
    before_answer: str,
    after_answer: str,
    generation_failed: bool,
    attempted_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Grounded question-conditioned safety gate with V5 structural-progress policy.

    The LLM does NOT choose KEEP/REVERT. It independently assesses BEFORE and
    AFTER answer entailment and supplies extractive evidence citations. Python
    validates those citations, checks primitive-level structural progress, and
    deterministically applies the final V5 harm-vs-progress gate policy.
    """
    structural_progress, structural_reason = _structural_progress(
        repair_label,
        attempted_history,
    )

    if generation_failed:
        return {
            "keep_change": False,
            "before_entailment": "unsupported",
            "after_entailment": "unsupported",
            "before_entailment_raw": "unsupported",
            "after_entailment_raw": "unsupported",
            "before_answers_question": bool(before_answer.strip()),
            "after_answers_question": False,
            "after_has_unsupported_claim": True,
            "new_conflict_introduced": False,
            "useful_evidence_lost": False,
            "before_support": [],
            "after_support": [],
            "invalid_before_support": [],
            "invalid_after_support": [],
            "structural_progress": structural_progress,
            "structural_progress_reason": structural_reason,
            "answer_changed": _answer_changed(before_answer, after_answer),
            "hard_harm_reasons": ["candidate_generation_failed"],
            "decision_policy": "generation_failed_revert",
            "support_improved": False,
            "accepted_progress": False,
            "reason": "The candidate answer generation failed.",
            "technical_error": "candidate_generation_failed",
            "gate_version": SAFE_GATE_VERSION,
        }

    before_evidence = _format_grounding_evidence(
        before_trace.get("retrieval_events", [])
    )
    after_evidence = _format_grounding_evidence(after_trace.get("retrieval_events", []))

    prompt = f"""Assess one repair step for QUESTION-CONDITIONED factual support.

You are an evidence verifier, not the repair controller. Do NOT output KEEP or REVERT.
Assess BEFORE and AFTER independently using only the supplied evidence.

Question:
{question}

Repair just attempted:
{repair_label}

BEFORE evidence (titles intentionally omitted):
{before_evidence}

BEFORE answer:
{before_answer}

AFTER evidence (titles intentionally omitted):
{after_evidence}

AFTER candidate answer:
{after_answer}

For BOTH BEFORE and AFTER:
- Decide whether the answer actually answers THIS question.
- Classify question-conditioned entailment as:
  * entailed: the evidence supports the answer AND the exact relation asked by the question;
  * partial: relevant support exists but one necessary link/relation is incomplete;
  * unsupported: the answer is not established by the evidence;
  * contradicted: the evidence directly supports an incompatible answer/relation.
- Provide one or more support citations when claiming entailed or partial.
- Each citation MUST contain the exact PASSAGE_ID and a SHORT VERBATIM quote copied from that passage.
- Do not cite a passage merely because the answer entity appears there. The quote must support the relation required by the question.

Critical relation rules:
- Verify words/relations such as other, same, before, after, first, larger, younger, nationality, location, creator, director, profession, comparison, and identity explicitly.
- Entity overlap alone is NOT question-conditioned support.
- If a passage says entity A is one group and entity B is the "other" group, answering A to a question asking for the other group is contradicted.
- Do not infer a fact because it appears in the question.
- Do not use passage titles as factual evidence.
- Do not assume access to a canonical/reference answer.
- A refusal, "not enough information", or a list of alternatives without selecting one does not answer the question.

AFTER-only harm fields:
- after_has_unsupported_claim=true if the AFTER answer adds a factual claim not supported by AFTER evidence.
- new_conflict_introduced=true only when the repair creates a new answer-relevant contradiction/conflict.
- useful_evidence_lost=true only when answer-relevant information available BEFORE is removed/corrupted/made materially less usable AFTER.

In reason, explain the question relation and why the AFTER answer is or is not grounded. Do not recommend KEEP/REVERT.
"""

    judge = llm.with_structured_output(
        SafeGroundingAssessment,
        method="json_schema",
    )

    try:
        assessment = invoke_structured(judge, prompt)

        valid_before, invalid_before = _validate_support_refs(
            list(assessment.before_support or []), before_trace
        )
        valid_after, invalid_after = _validate_support_refs(
            list(assessment.after_support or []), after_trace
        )

        before_raw = str(assessment.before_entailment)
        after_raw = str(assessment.after_entailment)
        before_answers = bool(assessment.before_answers_question)
        after_answers = bool(assessment.after_answers_question)

        before_entailment = _grounded_entailment(
            before_raw, before_answers, valid_before
        )
        after_entailment = _grounded_entailment(after_raw, after_answers, valid_after)

        before_score = SUPPORT_SCORE[before_entailment]
        after_score = SUPPORT_SCORE[after_entailment]
        answer_changed = _answer_changed(before_answer, after_answer)
        non_answer = _looks_like_non_answer(after_answer)

        # V5 delegates KEEP/REVERT to a pure deterministic policy.
        policy = _decide_v5_gate(
            answer_changed=answer_changed,
            before_entailment=before_entailment,
            after_entailment=after_entailment,
            before_score=before_score,
            after_score=after_score,
            after_answers=after_answers,
            non_answer=non_answer,
            unsupported_claim=bool(assessment.after_has_unsupported_claim),
            new_conflict=bool(assessment.new_conflict_introduced),
            useful_loss=bool(assessment.useful_evidence_lost),
            valid_after=valid_after,
            structural_progress=structural_progress,
        )
        keep_change = bool(policy["keep_change"])
        hard_harm_reasons = list(policy["hard_harm_reasons"])
        decision_policy = str(policy["decision_policy"])
        improvement = bool(policy["support_improved"])
        accepted_progress = bool(policy["accepted_progress"])

        return {
            "keep_change": keep_change,
            "before_entailment": before_entailment,
            "after_entailment": after_entailment,
            "before_entailment_raw": before_raw,
            "after_entailment_raw": after_raw,
            "before_answers_question": before_answers,
            "after_answers_question": after_answers,
            "after_has_unsupported_claim": bool(assessment.after_has_unsupported_claim),
            "new_conflict_introduced": bool(assessment.new_conflict_introduced),
            "useful_evidence_lost": bool(assessment.useful_evidence_lost),
            "before_support": valid_before,
            "after_support": valid_after,
            "invalid_before_support": invalid_before,
            "invalid_after_support": invalid_after,
            "structural_progress": structural_progress,
            "structural_progress_reason": structural_reason,
            "answer_changed": answer_changed,
            "hard_harm_reasons": hard_harm_reasons,
            "decision_policy": decision_policy,
            "support_improved": improvement,
            "accepted_progress": accepted_progress,
            "reason": str(assessment.reason or ""),
            "technical_error": "",
            "gate_version": SAFE_GATE_VERSION,
        }
    except Exception as error:
        return {
            "keep_change": False,
            "before_entailment": "unsupported",
            "after_entailment": "unsupported",
            "before_entailment_raw": "unsupported",
            "after_entailment_raw": "unsupported",
            "before_answers_question": bool(before_answer.strip()),
            "after_answers_question": False,
            "after_has_unsupported_claim": True,
            "new_conflict_introduced": False,
            "useful_evidence_lost": False,
            "before_support": [],
            "after_support": [],
            "invalid_before_support": [],
            "invalid_after_support": [],
            "structural_progress": structural_progress,
            "structural_progress_reason": structural_reason,
            "answer_changed": _answer_changed(before_answer, after_answer),
            "hard_harm_reasons": ["safety_judge_failed"],
            "decision_policy": "judge_failure_revert",
            "support_improved": False,
            "accepted_progress": False,
            "reason": "The grounded safety assessment failed, so the repair was undone.",
            "technical_error": f"safety_judge_failed:{type(error).__name__}",
            "gate_version": SAFE_GATE_VERSION,
        }


def _attach_safe_decision(
    trace: Dict[str, Any],
    failure: str,
    decision: Dict[str, Any],
) -> None:
    entry = next(
        (
            item
            for item in reversed(trace.get("repair_history", []))
            if item.get("failure") == failure
        ),
        None,
    )
    if entry is not None:
        entry["safe_composer_decision"] = deepcopy(decision)
        entry["reverted"] = False


def _prepare_for_primitive(
    trace: Dict[str, Any],
    current_answer: str,
) -> Dict[str, Any]:
    """Give a later primitive the answer produced by the current safe state.

    L repair uses failure_answer/final_answer as observable context. During
    sequential safe composition it must see the answer from the most recently
    kept state, not the stale answer from the originally corrupted trace.
    """
    working = deepcopy(trace)
    working["failure_answer"] = current_answer
    working["final_answer"] = current_answer
    return working


def _restore_failure_stage_fields(
    candidate: Dict[str, Any],
    original: Dict[str, Any],
) -> None:
    """Preserve the original pre-repair answer fields for final evaluation."""
    candidate["failure_answer"] = original.get("failure_answer", "")
    if "failure_evaluation" in original:
        candidate["failure_evaluation"] = deepcopy(original["failure_evaluation"])


def _finalize_safe_trace(
    original_trace: Dict[str, Any],
    repaired_trace: Dict[str, Any],
    final_safe_answer: str,
    repair_order: List[str],
    experiment_label: str,
) -> Dict[str, Any]:
    """Finalize Safe-v5 without generating a new, un-gated answer.

    In v1, each evidence-changing step was judged using a candidate answer, but
    the shared fixed/reverse finalizer then generated a fresh answer after the
    gate. That meant the final answer had never been safety-gated. Safe-v5 evaluates
    the last accepted safe-state answer directly. Fixed/reverse code is untouched.
    """
    original = deepcopy(original_trace)
    repaired = deepcopy(repaired_trace)
    failures = list(original.get("true_failures", []))

    evidence_changed = _evidence_signature(original) != _evidence_signature(repaired)
    final_answer = str(final_safe_answer or "")

    repair_evaluation = evaluate_prediction(
        repaired["question"],
        final_answer,
        repaired["canonical_answer"],
        generation_failed=False,
    )

    repaired["final_answer"] = final_answer

    failure_after, diagnosis = diagnose_failures_after_repair(
        original,
        repaired,
        repair_evaluation,
    )

    outcome_indeterminate = bool(
        repair_evaluation.get("generation_failed", False)
        or repair_evaluation.get("semantic_judge_error", False)
    )
    if outcome_indeterminate:
        failure_after = list(dict.fromkeys([*failure_after, *failures]))

    diagnosis["outcome_indeterminate"] = outcome_indeterminate
    diagnosis["evidence_changed_by_repair"] = evidence_changed
    diagnosis["final_generation_skipped"] = True
    diagnosis["final_answer_source"] = "last_accepted_safe_state"

    original_failures = set(failures)
    new_failures = [
        failure for failure in failure_after if failure not in original_failures
    ]

    before_evaluation = original.get("failure_evaluation") or original.get(
        "evaluation", {}
    )
    before_semantic_evaluable = bool(
        not before_evaluation.get("generation_failed", False)
        and not before_evaluation.get("semantic_judge_error", False)
    )
    after_semantic_evaluable = bool(
        not repair_evaluation.get("generation_failed", False)
        and not repair_evaluation.get("semantic_judge_error", False)
    )
    semantic_regression = bool(
        before_semantic_evaluable
        and after_semantic_evaluable
        and before_evaluation.get("semantic_correct", False)
        and not repair_evaluation.get("semantic_correct", False)
    )

    structural_regression = any(failure in {"M", "D"} for failure in new_failures)
    regression_detected = bool(semantic_regression or structural_regression)

    diagnosis["semantic_regression_detected"] = semantic_regression
    diagnosis["structural_regression_detected"] = structural_regression

    repaired["experiment_stage"] = "compound_repair_safe"
    repaired["repair_type"] = "+".join(failures)
    repaired["repair_mode"] = "safe"
    repaired["repair_order"] = list(repair_order)
    repaired["repair_experiment"] = experiment_label
    repaired["repair_evaluation"] = repair_evaluation
    repaired["evaluation"] = repair_evaluation
    repaired["failure_after_repair"] = failure_after
    repaired["new_failures_after_repair"] = new_failures
    repaired["semantic_regression_detected"] = semantic_regression
    repaired["structural_regression_detected"] = structural_regression
    repaired["regression_detected"] = regression_detected
    repaired["repair_diagnosis"] = diagnosis
    repaired["model_manifest"] = model_manifest()
    repaired["repair_evidence_changed"] = evidence_changed
    repaired["repair_generation_skipped"] = True
    repaired["safe_final_answer_reused"] = True
    repaired["safe_final_answer_source"] = "last_accepted_safe_state"

    repaired["answer_claims"] = (
        [
            {
                "claim": final_answer,
                "source": f"repair_{experiment_label}",
                "evidence_ids": [
                    item.get("passage_id", "")
                    for item in repaired.get("retrieval_events", [])
                ],
            }
        ]
        if final_answer
        else []
    )

    # No extra final answer-generation call is made in Safe-v5. Candidate
    # generation occurred inside the gated steps and is recorded there.
    repaired["repair_latency_ms"] = 0
    repaired["repair_token_usage"] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    return repaired


def _run_one_safe_trace(
    source_trace: Dict[str, Any],
    index: int,
    total_count: int,
    experiment_label: str,
) -> Dict[str, Any]:
    """Run the existing Safe-v5 logic for exactly one trace.

    This function contains the same planner/repair/gate/finalization behavior as
    the original loop. It is executed in a worker process so the parent can
    enforce a real wall-clock timeout for the whole trace on Windows.
    """
    trace_start = time.perf_counter()
    trace_id = str(source_trace.get("trace_id", ""))
    original = deepcopy(source_trace)
    current = deepcopy(original)

    _log(
        f"[{experiment_label}] trace {index}/{total_count} "
        f"({trace_id or 'no-trace-id'}) | planner START"
    )
    planner_start = time.perf_counter()
    plan = _choose_repair_order(current)
    order = list(plan["repair_order"])
    _log(
        f"[{experiment_label}] trace {index}/{total_count} | planner DONE "
        f"in {time.perf_counter() - planner_start:.1f}s | order={order}"
    )

    current_answer = _observed_answer(original)
    steps = []
    no_change_steps = []
    attempted_repairs = set()
    attempted_order: List[str] = []

    # Acyclic one-pass composer: each primitive can be attempted at most once.
    for failure in order:
        if failure in attempted_repairs:
            continue
        if len(attempted_repairs) >= MAX_SAFE_STEPS:
            break

        attempted_repairs.add(failure)
        attempted_order.append(failure)

        before = deepcopy(current)
        before_answer = current_answer
        history_count_before = len(before.get("repair_history", []))

        _log(
            f"[{experiment_label}] trace {index}/{total_count} | "
            f"{failure} repair START"
        )
        repair_start = time.perf_counter()

        try:
            working = _prepare_for_primitive(current, current_answer)
            candidate = apply_repair(working, failure)
            _restore_failure_stage_fields(candidate, original)
        except Exception as error:
            technical_error = f"repair_failed:{type(error).__name__}:{error}"
            current = before
            current.setdefault("repair_history", []).append(
                {
                    "failure": failure,
                    "repair": "safe_composer_reverted_attempt",
                    "reverted": True,
                    "attempted_repair_history": [],
                    "safe_composer_decision": {
                        "keep_change": False,
                        "before_entailment": "unsupported",
                        "after_entailment": "unsupported",
                        "before_answers_question": bool(before_answer.strip()),
                        "after_answers_question": False,
                        "after_has_unsupported_claim": True,
                        "new_conflict_introduced": False,
                        "useful_evidence_lost": False,
                        "hard_harm_reasons": ["repair_call_failed"],
                        "reason": "The repair call failed, so no change was kept.",
                        "technical_error": technical_error,
                        "gate_version": SAFE_GATE_VERSION,
                    },
                }
            )
            steps.append(
                {
                    "failure": failure,
                    "keep_change": False,
                    "reason": "The repair call failed, so no change was kept.",
                    "technical_error": technical_error,
                    "candidate_answer": "",
                    "candidate_generation_failed": True,
                    "evidence_changed": False,
                }
            )
            _log(
                f"[{experiment_label}] trace {index}/{total_count} | "
                f"{failure} repair ERROR after "
                f"{time.perf_counter() - repair_start:.1f}s | {technical_error}"
            )
            continue

        attempted_history = deepcopy(
            candidate.get("repair_history", [])[history_count_before:]
        )
        evidence_changed = _evidence_signature(before) != _evidence_signature(candidate)

        _log(
            f"[{experiment_label}] trace {index}/{total_count} | "
            f"{failure} repair DONE in "
            f"{time.perf_counter() - repair_start:.1f}s | "
            f"evidence_changed={evidence_changed}"
        )

        # IMPORTANT runtime fix: a metadata-only/no-op primitive must not cause
        # a fresh stochastic answer or an unnecessary safety-judge call.
        # Keep the primitive's diagnostic repair_history, preserve the current
        # answer, and move forward exactly once.
        if not evidence_changed:
            current = candidate
            current_answer = before_answer
            current["final_answer"] = current_answer
            no_change_steps.append(
                {
                    "failure": failure,
                    "reason": (
                        "Primitive produced no observable evidence change; "
                        "candidate generation and safety judge were skipped."
                    ),
                    "technical_error": "",
                    "evidence_changed": False,
                    "answer_reused": True,
                    "gate_version": SAFE_GATE_VERSION,
                }
            )
            _log(
                f"[{experiment_label}] trace {index}/{total_count} | "
                f"{failure} NO-OP -> reuse current answer; judge SKIPPED"
            )
            continue

        _log(
            f"[{experiment_label}] trace {index}/{total_count} | "
            f"{failure} candidate generation START"
        )
        generation_start = time.perf_counter()
        generation = generate_answer(
            candidate["question"],
            candidate.get("retrieval_events", []),
        )
        candidate_answer = str(generation.get("text", "") or "")
        generation_failed = bool(generation.get("generation_failed", False))
        _log(
            f"[{experiment_label}] trace {index}/{total_count} | "
            f"{failure} candidate generation DONE in "
            f"{time.perf_counter() - generation_start:.1f}s | "
            f"failed={generation_failed}"
        )

        _log(
            f"[{experiment_label}] trace {index}/{total_count} | "
            f"{failure} safety judge START"
        )
        judge_start = time.perf_counter()
        decision = _judge_step(
            candidate["question"],
            failure,
            before,
            candidate,
            before_answer,
            candidate_answer,
            generation_failed,
            attempted_history,
        )
        _log(
            f"[{experiment_label}] trace {index}/{total_count} | "
            f"{failure} safety judge DONE in "
            f"{time.perf_counter() - judge_start:.1f}s | "
            f"decision={'KEEP' if decision['keep_change'] else 'REVERT'}"
        )

        step_record = {
            "failure": failure,
            "keep_change": bool(decision["keep_change"]),
            "before_entailment": decision.get("before_entailment", ""),
            "after_entailment": decision.get("after_entailment", ""),
            "before_entailment_raw": decision.get("before_entailment_raw", ""),
            "after_entailment_raw": decision.get("after_entailment_raw", ""),
            "before_answers_question": bool(
                decision.get("before_answers_question", False)
            ),
            "after_answers_question": bool(
                decision.get("after_answers_question", False)
            ),
            "after_has_unsupported_claim": bool(
                decision.get("after_has_unsupported_claim", False)
            ),
            "new_conflict_introduced": bool(
                decision.get("new_conflict_introduced", False)
            ),
            "useful_evidence_lost": bool(decision.get("useful_evidence_lost", False)),
            "before_support": deepcopy(decision.get("before_support", [])),
            "after_support": deepcopy(decision.get("after_support", [])),
            "invalid_before_support": deepcopy(
                decision.get("invalid_before_support", [])
            ),
            "invalid_after_support": deepcopy(
                decision.get("invalid_after_support", [])
            ),
            "structural_progress": bool(decision.get("structural_progress", False)),
            "structural_progress_reason": decision.get(
                "structural_progress_reason", ""
            ),
            "answer_changed": bool(decision.get("answer_changed", False)),
            "hard_harm_reasons": list(decision.get("hard_harm_reasons", []) or []),
            "reason": decision["reason"],
            "technical_error": decision["technical_error"],
            "gate_version": decision.get("gate_version", SAFE_GATE_VERSION),
            "candidate_answer": candidate_answer,
            "candidate_generation_failed": generation_failed,
            "evidence_changed": True,
        }

        if decision["keep_change"]:
            current = candidate
            current_answer = candidate_answer
            current["final_answer"] = current_answer
            _attach_safe_decision(current, failure, decision)
        else:
            current = before
            current_answer = before_answer
            current.setdefault("repair_history", []).append(
                {
                    "failure": failure,
                    "repair": "safe_composer_reverted_attempt",
                    "reverted": True,
                    "attempted_repair_history": attempted_history,
                    "safe_composer_decision": deepcopy(decision),
                }
            )

        steps.append(step_record)

    current["safe_composer_history"] = {
        "safe_composer_version": SAFE_COMPOSER_VERSION,
        "planner_version": plan.get("planner_version", SAFE_PLANNER_VERSION),
        "gate_version": SAFE_GATE_VERSION,
        "available_repairs": _available_repairs(original),
        "planner_raw_order": plan.get("raw_repair_order", []),
        "planned_order": order,
        "attempted_order": attempted_order,
        "planner_reason": plan["reason"],
        "planner_technical_error": plan["technical_error"],
        # 'steps' intentionally contains only evidence-changing attempts (or
        # true repair errors). This keeps analyzer KEEP/REVERT counts honest.
        "steps": steps,
        "no_change_steps": no_change_steps,
        "no_change_step_count": len(no_change_steps),
        "bounded_one_pass": True,
        "max_safe_steps": MAX_SAFE_STEPS,
    }

    _log(f"[{experiment_label}] trace {index}/{total_count} | finalization START")
    finalization_start = time.perf_counter()
    repaired = _finalize_safe_trace(
        original,
        current,
        current_answer,
        order,
        experiment_label,
    )
    _log(
        f"[{experiment_label}] trace {index}/{total_count} | "
        f"finalization DONE in "
        f"{time.perf_counter() - finalization_start:.1f}s"
    )

    return repaired


def run_safe(input_path: str, output_path: str, experiment_label: str) -> None:
    with open(input_path, "r", encoding="utf-8") as file:
        traces = json.load(file)

    if not isinstance(traces, list):
        raise ValueError(f"Expected a JSON list in {input_path}.")

    # Resume by default. Set COMPOREPAIR_SAFE_FORCE_RESTART=1 to intentionally
    # discard the checkpoint and rerun this condition from the beginning.
    if _force_restart_requested():
        existing_results: List[Dict[str, Any]] = []
        if os.path.exists(output_path):
            _log(
                f"[{experiment_label}] FORCE RESTART requested; existing "
                f"checkpoint will be overwritten."
            )
    else:
        existing_results = _load_checkpoint(output_path)

    results = list(existing_results)
    completed_trace_ids = {
        str(item.get("trace_id", ""))
        for item in existing_results
        if item.get("trace_id")
    }

    if existing_results:
        _log(
            f"[{experiment_label}] Resuming from checkpoint: "
            f"{len(existing_results)} completed trace(s)."
        )

    # Use one persistent spawned worker. On timeout, terminate that worker and
    # immediately create a fresh one so a blocking LLM/repair call cannot hold
    # the whole experiment forever.
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=1)

    try:
        for index, source_trace in enumerate(traces, start=1):
            trace_id = str(source_trace.get("trace_id", ""))
            if trace_id and trace_id in completed_trace_ids:
                _log(
                    f"[{experiment_label}] trace {index}/{len(traces)} "
                    f"({trace_id}) already completed; skipping."
                )
                continue

            trace_start = time.perf_counter()
            job = pool.apply_async(
                _run_one_safe_trace,
                (source_trace, index, len(traces), experiment_label),
            )

            try:
                repaired = job.get(timeout=MAX_TRACE_SECONDS)
            except mp.TimeoutError:
                _log(
                    f"[{experiment_label}] trace {index}/{len(traces)} "
                    f"({trace_id or 'no-trace-id'}) exceeded "
                    f"{MAX_TRACE_SECONDS:.0f}s; TERMINATING and SKIPPING trace."
                )
                pool.terminate()
                pool.join()
                pool = ctx.Pool(processes=1)
                continue

            results.append(repaired)
            if trace_id:
                completed_trace_ids.add(trace_id)

            # Save after every completed trace. os.replace makes the checkpoint
            # atomic, so Ctrl+C/power loss cannot leave a half-written JSON file.
            _atomic_save_results(results, output_path)

            _log(
                f"[{experiment_label}] processed trace {index}/{len(traces)} | "
                f"checkpoint saved | trace time="
                f"{time.perf_counter() - trace_start:.1f}s"
            )
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    # Final write is intentionally redundant; it guarantees the final path is
    # present even for an empty input and keeps the previous behavior explicit.
    _atomic_save_results(results, output_path)
    _log(f"Saved safe repair results: {output_path}")


def main():
    # Optional engineering convenience: run one condition at a time without
    # changing the default experiment set. Example:
    # COMPOREPAIR_SAFE_EXPERIMENT=M_D_L_safe python -m comporepair.run_safe
    selected = os.getenv("COMPOREPAIR_SAFE_EXPERIMENT", "").strip()
    experiments = EXPERIMENTS
    if selected:
        experiments = [item for item in EXPERIMENTS if item[0] == selected]
        if not experiments:
            valid = ", ".join(item[0] for item in EXPERIMENTS)
            raise ValueError(
                f"Unknown COMPOREPAIR_SAFE_EXPERIMENT={selected!r}. "
                f"Choose one of: {valid}"
            )

    for label, input_path, output_path in experiments:
        run_safe(input_path, output_path, label)


if __name__ == "__main__":
    main()
