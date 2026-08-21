import json
import multiprocessing as mp
import os
import re
import time
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .pipeline.baseline_rag import (
    evaluate_prediction,
    format_evidence,
    generate_answer,
    hash_text,
    invoke_structured,
    invoke_text,
    llm,
    model_manifest,
)
from .retrieval.vector_store import (
    REPAIR_CANDIDATE_K,
    RETRIEVAL_K,
    get_retriever,
)


TRACE_TIMEOUT_SECONDS = 180.0
B2_VERSION = "b2_generic_self_reflection_v1"
B3_VERSION = "b3_one_shot_joint_repair_v1"

COMPOUND_INPUTS = {
    "M_D": "data/results/compound/M_D_results.json",
    "M_L": "data/results/compound/M_L_results.json",
    "D_L": "data/results/compound/D_L_results.json",
    "M_D_L": "data/results/compound/M_D_L_results.json",
}

EXPERIMENTS = [
    (
        f"{condition}_b2",
        "b2",
        input_path,
        f"data/repaired_result/{condition}_b2_repair_results.json",
    )
    for condition, input_path in COMPOUND_INPUTS.items()
] + [
    (
        f"{condition}_b3",
        "b3",
        input_path,
        f"data/repaired_result/{condition}_b3_repair_results.json",
    )
    for condition, input_path in COMPOUND_INPUTS.items()
]


B2_SYSTEM_PROMPT = """You are performing a generic self-reflection/revision baseline for retrieval-augmented question answering.

Use only the current supplied evidence and the current answer.
Do not use outside knowledge or memory.
Do not retrieve new evidence.
Do not propose or execute evidence-repair operations.
Do not classify the problem into predefined failure types.
Do not mention any failure taxonomy.

Reconsider whether the current answer is actually supported by the supplied evidence and whether the evidence should be combined differently. Revise the answer only if necessary.

Return only the concise final answer.
Do not explain or show reasoning.
Normally return 1-5 words.
"""


B3_SYSTEM_PROMPT = """You are a generic one-shot joint evidence-repair planner for retrieval-augmented question answering.

You receive a question, the current evidence, and the current answer. Make ONE joint repair decision for the whole current state. You do not know or use any predefined failure taxonomy or specialized repair module.

Allowed actions in this single plan:
1. Remove at most one passage only if it is clearly misleading, conflicting, or harmful for answering the question.
2. Request at most one retrieval query if important or clarifying evidence may be missing.
3. Rewrite at most one passage only to clarify an unclear identity/reference/link using facts directly supported by the evidence already shown. The rewrite must preserve unrelated facts and must not invent new facts.
4. These generic actions may be combined in the same one-shot plan.
5. If no safe improvement is justified, set repair_needed=false and request no actions.

Strict rules:
- Use only neutral passage labels such as P1, P2, P3.
- Do not use outside knowledge or memory.
- Do not infer hidden passage IDs, titles, support labels, gold answers, or injected-failure metadata.
- Do not name or classify failures using a taxonomy.
- Do not put an unsupported guessed answer into the retrieval query.
- If both removal and rewrite are requested, choose different passage labels.
- rewritten_passage_text must be the complete replacement text for that one passage.
- The executor will perform the requested actions once; there is no second planning round.
"""


class JointRepairDecision(BaseModel):
    repair_needed: bool
    remove_passage_label: Optional[str] = None
    retrieval_needed: bool = False
    retrieval_query: Optional[str] = None
    rewrite_passage_label: Optional[str] = None
    rewritten_passage_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


B3_DECISION_SCHEMA_DESCRIPTION = """Return exactly one structured joint repair decision with these fields:
- repair_needed: boolean
- remove_passage_label: one label such as P4, or null
- retrieval_needed: boolean
- retrieval_query: one short retrieval query, or null
- rewrite_passage_label: one label such as P7, or null
- rewritten_passage_text: complete replacement text for that passage, or null
- confidence: number from 0 to 1
- reason: short logging reason
"""


def _timeout_output_path(output_path: str) -> str:
    name = os.path.splitext(os.path.basename(output_path))[0] + "_timeouts.json"
    return os.path.join("data", "timeouts", "repair", name)


def _failure_answer(trace: Dict[str, Any]) -> str:
    return str(
        trace.get("failure_answer")
        or trace.get("final_answer")
        or trace.get("baseline_answer")
        or ""
    )


def _failure_evaluation(trace: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(trace.get("failure_evaluation") or trace.get("evaluation", {}))


def _supporting_ids(trace: Dict[str, Any]) -> set:
    explicit = {
        str(item) for item in trace.get("gold_supporting_passage_ids", []) if item
    }
    if explicit:
        return explicit
    return {
        str(item.get("passage_id", ""))
        for item in trace.get("selected_evidence", [])
        if item.get("is_supporting", False) and item.get("passage_id")
    }


def _injected_missing_ids(trace: Dict[str, Any]) -> set:
    return {
        str(passage_id)
        for item in trace.get("failure_history", [])
        if item.get("failure") == "M"
        for passage_id in item.get("removed_passage_ids", [])
        if passage_id
    }


def _injected_distractor_ids(trace: Dict[str, Any]) -> set:
    return {
        str(item.get("distractor_passage_id", ""))
        for item in trace.get("failure_history", [])
        if item.get("failure") == "D" and item.get("distractor_passage_id")
    }


def _link_history(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        item for item in trace.get("failure_history", []) if item.get("failure") == "L"
    ]


def _diagnose_failures_after_repair(
    original_trace: Dict[str, Any],
    repaired_trace: Dict[str, Any],
    repair_evaluation: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Offline diagnosis only; gold/injection metadata never enters B2/B3 prompts."""
    final_documents = repaired_trace.get("retrieval_events", [])
    final_ids = {
        str(document.get("passage_id", ""))
        for document in final_documents
        if document.get("passage_id")
    }
    final_text_by_id = {
        str(document.get("passage_id", "")): str(document.get("text", ""))
        for document in final_documents
        if document.get("passage_id")
    }

    injected_missing_ids = _injected_missing_ids(original_trace)
    remaining_injected_missing_ids = injected_missing_ids - final_ids

    required_support = _supporting_ids(original_trace)
    missing_required_support_ids = required_support - final_ids
    support_complete_after = not missing_required_support_ids
    m_after = bool(missing_required_support_ids)

    known_distractors = _injected_distractor_ids(original_trace)
    remaining_injected_distractors = known_distractors & final_ids
    d_after = bool(remaining_injected_distractors)

    link_entries = _link_history(original_trace)
    remaining_link_targets: List[str] = []
    changed_link_targets: List[str] = []
    missing_link_targets: List[str] = []

    for entry in link_entries:
        target_id = str(entry.get("target_passage_id", ""))
        corrupted_text = str(entry.get("corrupted_passage_text", ""))
        if not target_id:
            continue
        if target_id not in final_ids:
            missing_link_targets.append(target_id)
            continue

        current_text = final_text_by_id.get(target_id, "")
        if current_text == corrupted_text:
            remaining_link_targets.append(target_id)
        else:
            changed_link_targets.append(target_id)

    l_after = bool(remaining_link_targets)

    failures: List[str] = []
    if m_after:
        failures.append("M")
    if d_after:
        failures.append("D")
    if l_after:
        failures.append("L")

    diagnosis = {
        "injected_missing_passage_ids": sorted(injected_missing_ids),
        "remaining_injected_missing_passage_ids": sorted(
            remaining_injected_missing_ids
        ),
        "required_supporting_passage_ids": sorted(required_support),
        "missing_required_supporting_passage_ids": sorted(
            missing_required_support_ids
        ),
        "final_passage_ids": sorted(final_ids),
        "known_distractor_ids": sorted(known_distractors),
        "remaining_injected_distractor_ids": sorted(remaining_injected_distractors),
        "injected_link_target_ids": sorted(
            str(item.get("target_passage_id", ""))
            for item in link_entries
            if item.get("target_passage_id")
        ),
        "remaining_injected_link_target_ids": sorted(remaining_link_targets),
        "changed_link_target_ids": sorted(changed_link_targets),
        "missing_link_target_ids": sorted(missing_link_targets),
        "M_present": m_after,
        "D_present": d_after,
        "L_present": l_after,
        "M_diagnosis_basis": "gold_support_coverage",
        "D_diagnosis_basis": "injected_distractor_presence",
        "L_diagnosis_basis": "injected_corrupted_text_still_present",
        "support_complete_after": support_complete_after,
        "generation_failed": bool(repair_evaluation.get("generation_failed", False)),
        "semantic_judge_error": bool(
            repair_evaluation.get("semantic_judge_error", False)
        ),
    }
    return failures, diagnosis


def _semantic_regression(
    before_evaluation: Dict[str, Any],
    after_evaluation: Dict[str, Any],
) -> bool:
    before_evaluable = bool(
        not before_evaluation.get("generation_failed", False)
        and not before_evaluation.get("semantic_judge_error", False)
    )
    after_evaluable = bool(
        not after_evaluation.get("generation_failed", False)
        and not after_evaluation.get("semantic_judge_error", False)
    )
    return bool(
        before_evaluable
        and after_evaluable
        and before_evaluation.get("semantic_correct", False)
        and not after_evaluation.get("semantic_correct", False)
    )


def _sum_token_usage(*usages: Dict[str, Any]) -> Dict[str, int]:
    result = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        result["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        result["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        result["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    return result


def _evidence_signature(trace: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        (
            str(item.get("passage_id", "")),
            str(item.get("text", "")),
        )
        for item in trace.get("retrieval_events", [])
    )


def _candidate_to_dict(document: Any) -> Dict[str, Any]:
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


def _normalise_passage_label(value: Optional[str], count: int) -> Optional[str]:
    text = str(value or "").strip().upper()
    match = re.fullmatch(r"\[?P(\d+)\]?", text)
    if not match:
        return None
    number = int(match.group(1))
    if number < 1 or number > count:
        return None
    return f"P{number}"


def _rank_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = [deepcopy(document) for document in documents]
    for rank, document in enumerate(ranked, start=1):
        document["rank"] = rank
    return ranked


def _finalize_baseline_trace(
    original_trace: Dict[str, Any],
    repaired_trace: Dict[str, Any],
    final_answer: str,
    repair_evaluation: Dict[str, Any],
    method: Literal["b2", "b3"],
    experiment_label: str,
    prompt_hashes_to_add: Dict[str, str],
    repair_latency_ms: int,
    repair_token_usage: Dict[str, int],
    method_history: Dict[str, Any],
    generation_skipped: bool,
) -> Dict[str, Any]:
    original = deepcopy(original_trace)
    repaired = deepcopy(repaired_trace)
    failures = list(original.get("true_failures", []))

    repaired["final_answer"] = str(final_answer or "")

    failure_after, diagnosis = _diagnose_failures_after_repair(
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

    evidence_changed = _evidence_signature(original) != _evidence_signature(repaired)
    diagnosis["outcome_indeterminate"] = outcome_indeterminate
    diagnosis["evidence_changed_by_repair"] = evidence_changed
    diagnosis["final_generation_skipped"] = generation_skipped

    original_failures = set(failures)
    new_failures = [
        failure for failure in failure_after if failure not in original_failures
    ]

    before_evaluation = _failure_evaluation(original)
    semantic_regression = _semantic_regression(before_evaluation, repair_evaluation)
    structural_regression = any(failure in {"M", "D"} for failure in new_failures)
    regression_detected = bool(semantic_regression or structural_regression)

    diagnosis["semantic_regression_detected"] = semantic_regression
    diagnosis["structural_regression_detected"] = structural_regression

    repaired["experiment_stage"] = f"compound_repair_{method}"
    repaired["repair_type"] = "+".join(failures)
    repaired["repair_mode"] = method
    repaired["repair_order"] = []
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
    repaired["repair_generation_skipped"] = generation_skipped
    repaired["repair_latency_ms"] = int(repair_latency_ms)
    repaired["repair_token_usage"] = repair_token_usage

    repaired["answer_claims"] = (
        [
            {
                "claim": str(final_answer or ""),
                "source": f"repair_{experiment_label}",
                "evidence_ids": [
                    item.get("passage_id", "")
                    for item in repaired.get("retrieval_events", [])
                ],
            }
        ]
        if str(final_answer or "")
        else []
    )

    prompt_hashes = dict(repaired.get("prompt_hashes", {}))
    prompt_hashes.update(prompt_hashes_to_add)
    repaired["prompt_hashes"] = prompt_hashes

    history = list(repaired.get("repair_history", []))
    history.append(method_history)
    repaired["repair_history"] = history
    repaired["baseline_method_history"] = method_history

    return repaired


def process_b2_trace(
    source_trace: Dict[str, Any],
    experiment_label: str,
) -> Dict[str, Any]:
    original = deepcopy(source_trace)
    repaired = deepcopy(original)

    evidence_text = format_evidence(original.get("retrieval_events", []))
    current_answer = _failure_answer(original)

    messages = [
        SystemMessage(content=B2_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question:\n{original['question']}\n\n"
                f"Current evidence:\n{evidence_text}\n\n"
                f"Current answer:\n{current_answer}\n\n"
                "Return the revised final answer:"
            )
        ),
    ]

    revision = invoke_text(messages)
    final_answer = str(revision.get("text", "") or "")
    repair_evaluation = evaluate_prediction(
        original["question"],
        final_answer,
        original["canonical_answer"],
        generation_failed=bool(revision.get("generation_failed", False)),
    )

    prompt_hash = hash_text(
        f"{B2_SYSTEM_PROMPT}\nQuestion:\n{{question}}\n"
        "Current evidence:\n{evidence}\nCurrent answer:\n{failure_answer}"
    )

    method_history = {
        "baseline": "B2",
        "method": "generic_self_reflection_revision",
        "version": B2_VERSION,
        "failure_labels_exposed": False,
        "specialized_repairs_used": False,
        "retrieval_performed": False,
        "evidence_modified": False,
        "current_answer_provided": True,
        "generation_attempts": int(revision.get("attempts", 0) or 0),
        "technical_error": bool(revision.get("generation_failed", False)),
    }

    return _finalize_baseline_trace(
        original_trace=original,
        repaired_trace=repaired,
        final_answer=final_answer,
        repair_evaluation=repair_evaluation,
        method="b2",
        experiment_label=experiment_label,
        prompt_hashes_to_add={"b2_self_reflection": prompt_hash},
        repair_latency_ms=int(revision.get("latency_ms", 0) or 0),
        repair_token_usage=revision.get("token_usage", {}),
        method_history=method_history,
        generation_skipped=False,
    )


def _plan_b3(trace: Dict[str, Any]) -> Tuple[JointRepairDecision, int]:
    evidence_text = format_evidence(trace.get("retrieval_events", []))
    current_answer = _failure_answer(trace)

    messages = [
        SystemMessage(content=B3_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question:\n{trace['question']}\n\n"
                f"Current evidence:\n{evidence_text}\n\n"
                f"Current answer:\n{current_answer}\n\n"
                f"{B3_DECISION_SCHEMA_DESCRIPTION}"
            )
        ),
    ]

    runnable = llm.with_structured_output(
        JointRepairDecision,
        method="json_schema",
    )
    start = time.perf_counter()
    decision = invoke_structured(runnable, messages)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return decision, latency_ms


def _execute_b3_plan(
    trace: Dict[str, Any],
    decision: JointRepairDecision,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    repaired = deepcopy(trace)
    original_documents = [
        deepcopy(document) for document in repaired.get("retrieval_events", [])
    ]
    labelled_documents = [
        {"label": f"P{index}", "document": deepcopy(document)}
        for index, document in enumerate(original_documents, start=1)
    ]

    remove_label = _normalise_passage_label(
        decision.remove_passage_label,
        len(labelled_documents),
    )
    rewrite_label = _normalise_passage_label(
        decision.rewrite_passage_label,
        len(labelled_documents),
    )

    audit: Dict[str, Any] = {
        "repair_needed": bool(decision.repair_needed),
        "requested_remove_passage_label": decision.remove_passage_label,
        "requested_retrieval": bool(decision.retrieval_needed),
        "requested_retrieval_query": str(decision.retrieval_query or ""),
        "requested_rewrite_passage_label": decision.rewrite_passage_label,
        "confidence": float(decision.confidence),
        "reason": str(decision.reason or ""),
        "remove_applied": False,
        "removed_passage_label": None,
        "removed_passage_id": None,
        "rewrite_applied": False,
        "rewritten_passage_label": None,
        "rewritten_passage_id": None,
        "retrieval_performed": False,
        "retrieval_query": "",
        "retrieval_k": REPAIR_CANDIDATE_K,
        "retrieved_candidate_ids": [],
        "added_passage_ids": [],
        "invalid_action_reasons": [],
    }

    if not decision.repair_needed:
        audit["decision"] = "no_repair"
        repaired["retrieval_events"] = _rank_documents(original_documents)
        return repaired, audit

    if decision.remove_passage_label and remove_label is None:
        audit["invalid_action_reasons"].append("invalid_remove_passage_label")
    if decision.rewrite_passage_label and rewrite_label is None:
        audit["invalid_action_reasons"].append("invalid_rewrite_passage_label")

    if remove_label is not None:
        for item in labelled_documents:
            if item["label"] == remove_label:
                audit["remove_applied"] = True
                audit["removed_passage_label"] = remove_label
                audit["removed_passage_id"] = str(
                    item["document"].get("passage_id", "") or ""
                )
                break
        labelled_documents = [
            item for item in labelled_documents if item["label"] != remove_label
        ]

    rewrite_text = str(decision.rewritten_passage_text or "").strip()
    if rewrite_label is not None:
        if rewrite_label == remove_label:
            audit["invalid_action_reasons"].append(
                "rewrite_target_same_as_removed_passage"
            )
        elif not rewrite_text:
            audit["invalid_action_reasons"].append("empty_rewritten_passage_text")
        else:
            for item in labelled_documents:
                if item["label"] != rewrite_label:
                    continue
                original_text = str(item["document"].get("text", "") or "")
                if rewrite_text == original_text:
                    audit["invalid_action_reasons"].append(
                        "rewritten_text_identical_to_original"
                    )
                    break
                item["document"]["text"] = rewrite_text
                audit["rewrite_applied"] = True
                audit["rewritten_passage_label"] = rewrite_label
                audit["rewritten_passage_id"] = str(
                    item["document"].get("passage_id", "") or ""
                )
                break

    documents = [deepcopy(item["document"]) for item in labelled_documents]
    documents = documents[:RETRIEVAL_K]

    if decision.retrieval_needed:
        retrieval_query = str(decision.retrieval_query or "").strip()
        if not retrieval_query:
            audit["invalid_action_reasons"].append("empty_retrieval_query")
        else:
            audit["retrieval_performed"] = True
            audit["retrieval_query"] = retrieval_query
            candidates = get_retriever(k=REPAIR_CANDIDATE_K).invoke(retrieval_query)
            selected_ids = {
                str(document.get("passage_id", ""))
                for document in documents
                if document.get("passage_id")
            }

            for candidate in candidates:
                candidate_dict = _candidate_to_dict(candidate)
                candidate_id = str(candidate_dict.get("passage_id", "") or "")
                if candidate_id:
                    audit["retrieved_candidate_ids"].append(candidate_id)

                if len(documents) >= RETRIEVAL_K:
                    continue
                if not candidate_id or candidate_id in selected_ids:
                    continue

                documents.append(candidate_dict)
                selected_ids.add(candidate_id)
                audit["added_passage_ids"].append(candidate_id)

    repaired["retrieval_events"] = _rank_documents(documents)
    audit["context_count_before"] = len(original_documents)
    audit["context_count_after"] = len(repaired["retrieval_events"])
    audit["evidence_changed"] = (
        _evidence_signature(trace) != _evidence_signature(repaired)
    )
    audit["decision"] = (
        "joint_repair_executed" if audit["evidence_changed"] else "no_effective_change"
    )
    return repaired, audit


def process_b3_trace(
    source_trace: Dict[str, Any],
    experiment_label: str,
) -> Dict[str, Any]:
    original = deepcopy(source_trace)

    decision, planner_latency_ms = _plan_b3(original)
    repaired, execution = _execute_b3_plan(original, decision)
    evidence_changed = _evidence_signature(original) != _evidence_signature(repaired)

    generation: Dict[str, Any] = {}
    if evidence_changed:
        generation = generate_answer(
            repaired["question"],
            repaired.get("retrieval_events", []),
        )
        final_answer = str(generation.get("text", "") or "")
        repair_evaluation = evaluate_prediction(
            repaired["question"],
            final_answer,
            repaired["canonical_answer"],
            generation_failed=bool(generation.get("generation_failed", False)),
        )
        generation_skipped = False
    else:
        final_answer = _failure_answer(original)
        repair_evaluation = _failure_evaluation(original)
        generation_skipped = True

    planner_prompt_hash = hash_text(
        f"{B3_SYSTEM_PROMPT}\nQuestion:\n{{question}}\n"
        "Current evidence:\n{evidence}\nCurrent answer:\n{failure_answer}\n"
        f"{B3_DECISION_SCHEMA_DESCRIPTION}"
    )

    prompt_hashes = {"b3_joint_planner": planner_prompt_hash}
    if generation:
        prompt_hashes["b3_final_generation"] = str(
            generation.get("prompt_hash", "") or ""
        )

    method_history = {
        "baseline": "B3",
        "method": "one_shot_joint_repair",
        "version": B3_VERSION,
        "failure_labels_exposed": False,
        "specialized_repairs_used": False,
        "single_joint_planner_call": True,
        "max_retrieval_rounds": 1,
        "max_passage_removals": 1,
        "max_passage_rewrites": 1,
        "planner_decision": decision.model_dump(),
        "execution": execution,
        "planner_latency_ms": planner_latency_ms,
        "final_generation_performed": not generation_skipped,
        "technical_error": bool(generation.get("generation_failed", False))
        if generation
        else False,
    }

    generation_latency_ms = int(generation.get("latency_ms", 0) or 0)
    generation_usage = generation.get("token_usage", {}) if generation else {}

    return _finalize_baseline_trace(
        original_trace=original,
        repaired_trace=repaired,
        final_answer=final_answer,
        repair_evaluation=repair_evaluation,
        method="b3",
        experiment_label=experiment_label,
        prompt_hashes_to_add=prompt_hashes,
        repair_latency_ms=planner_latency_ms + generation_latency_ms,
        repair_token_usage=_sum_token_usage(generation_usage),
        method_history=method_history,
        generation_skipped=generation_skipped,
    )


def process_single_trace(
    source_trace: Dict[str, Any],
    method: Literal["b2", "b3"],
    experiment_label: str,
) -> Dict[str, Any]:
    if method == "b2":
        return process_b2_trace(source_trace, experiment_label)
    if method == "b3":
        return process_b3_trace(source_trace, experiment_label)
    raise ValueError(f"Unsupported baseline method: {method}")


def run_experiment(
    input_path: str,
    output_path: str,
    method: Literal["b2", "b3"],
    experiment_label: str,
) -> None:
    with open(input_path, "r", encoding="utf-8") as file:
        traces = json.load(file)

    if not isinstance(traces, list):
        raise ValueError(f"Expected a JSON list in {input_path}.")

    results: List[Dict[str, Any]] = []
    timeouts: List[Dict[str, Any]] = []

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=1)

    try:
        for index, source_trace in enumerate(traces, start=1):
            job = pool.apply_async(
                process_single_trace,
                (source_trace, method, experiment_label),
            )

            try:
                repaired = job.get(timeout=TRACE_TIMEOUT_SECONDS)
            except mp.TimeoutError:
                timeout_trace = deepcopy(source_trace)
                timeout_trace["runtime"] = {
                    "timed_out": True,
                    "stage": experiment_label,
                    "timeout_seconds": TRACE_TIMEOUT_SECONDS,
                }
                timeouts.append(timeout_trace)
                print(
                    f"{experiment_label} skipped trace {index}/{len(traces)}: "
                    f"TIMEOUT (> {TRACE_TIMEOUT_SECONDS:.0f}s)"
                )
                pool.terminate()
                pool.join()
                pool = ctx.Pool(processes=1)
                continue
            except Exception as error:
                print(
                    f"{experiment_label} skipped trace {index}/{len(traces)}: "
                    f"ERROR ({error})"
                )
                continue

            results.append(repaired)
            print(f"{experiment_label} processed trace {index}/{len(traces)}")

    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    timeout_path = _timeout_output_path(output_path)
    os.makedirs(os.path.dirname(timeout_path), exist_ok=True)
    with open(timeout_path, "w", encoding="utf-8") as file:
        json.dump(timeouts, file, indent=2, ensure_ascii=False)

    print(f"Saved {method.upper()} results: {output_path}")
    print(f"Saved {method.upper()} timeouts: {timeout_path} ({len(timeouts)})")


def main() -> None:
    selected = os.getenv("COMPOREPAIR_B2_B3_EXPERIMENT", "").strip()
    experiments: Sequence[Tuple[str, str, str, str]] = EXPERIMENTS

    if selected:
        experiments = [item for item in EXPERIMENTS if item[0] == selected]
        if not experiments:
            valid = ", ".join(item[0] for item in EXPERIMENTS)
            raise ValueError(
                f"Unknown COMPOREPAIR_B2_B3_EXPERIMENT={selected!r}. "
                f"Choose one of: {valid}"
            )

    for label, method, input_path, output_path in experiments:
        run_experiment(
            input_path=input_path,
            output_path=output_path,
            method=method,  # type: ignore[arg-type]
            experiment_label=label,
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()