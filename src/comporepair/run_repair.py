import json
import os
import multiprocessing as mp
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

from .pipeline.baseline_rag import (
    evaluate_prediction,
    generate_answer,
    model_manifest,
)
from .repair.distractor import repair_distractor
from .repair.linkage import repair_linkage
from .repair.missing_evidence import repair_missing_evidence

TRACE_TIMEOUT_SECONDS = 180.0


def _timeout_output_path(output_path: str) -> str:
    name = os.path.splitext(os.path.basename(output_path))[0] + "_timeouts.json"
    return os.path.join("data", "timeouts", "repair", name)

EXPERIMENTS = [
    # (
    #     "M",
    #     "data/results/single/missing_evidence_results.json",
    #     "data/repaired_result/M_repair_results.json",
    #     ["M"],
    #     "fixed",
    # ),
    # (
    #     "D",
    #     "data/results/single/distractor_results.json",
    #     "data/repaired_result/D_repair_results.json",
    #     ["D"],
    #     "fixed",
    # ),
    # (
    #     "L",
    #     "data/results/single/linkage_results.json",
    #     "data/repaired_result/L_repair_results.json",
    #     ["L"],
    #     "fixed",
    # ),
    # (
    #     "M_D_fixed",
    #     "data/results/compound/M_D_results.json",
    #     "data/repaired_result/M_D_repair_results.json",
    #     ["M", "D"],
    #     "fixed",
    # ),
    # (
    #     "M_D_reverse",
    #     "data/results/compound/M_D_results.json",
    #     "data/repaired_result/M_D_reverse_repair_results.json",
    #     ["D", "M"],
    #     "reverse",
    # ),
    # (
    #     "M_L_fixed",
    #     "data/results/compound/M_L_results.json",
    #     "data/repaired_result/M_L_repair_results.json",
    #     ["M", "L"],
    #     "fixed",
    # ),
    # (
    #     "M_L_reverse",
    #     "data/results/compound/M_L_results.json",
    #     "data/repaired_result/M_L_reverse_repair_results.json",
    #     ["L", "M"],
    #     "reverse",
    # ),
    # (
    #     "D_L_fixed",
    #     "data/results/compound/D_L_results.json",
    #     "data/repaired_result/D_L_repair_results.json",
    #     ["D", "L"],
    #     "fixed",
    # ),
    # (
    #     "D_L_reverse",
    #     "data/results/compound/D_L_results.json",
    #     "data/repaired_result/D_L_reverse_repair_results.json",
    #     ["L", "D"],
    #     "reverse",
    # ),
    # (
    #     "M_D_L_fixed",
    #     "data/results/compound/M_D_L_results.json",
    #     "data/repaired_result/M_D_L_repair_results.json",
    #     ["M", "D", "L"],
    #     "fixed",
    # ),
    (
        "M_D_L_reverse",
        "data/results/compound/M_D_L_results.json",
        "data/repaired_result/M_D_L_reverse_repair_results.json",
        ["L", "D", "M"],
        "reverse",
    ),
]


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


def _evidence_signature(trace: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Stable evidence identity used only to decide whether regeneration is needed."""
    return tuple(
        (
            str(item.get("passage_id", "")),
            str(item.get("text", "")),
        )
        for item in trace.get("retrieval_events", [])
    )


def apply_repair(trace: Dict[str, Any], failure: str) -> Dict[str, Any]:
    if failure == "M":
        return repair_missing_evidence(trace)
    if failure == "D":
        return repair_distractor(trace)
    if failure == "L":
        return repair_linkage(trace)
    raise ValueError(f"Unsupported repair type: {failure}")


def diagnose_failures_after_repair(
    original_trace: Dict[str, Any],
    repaired_trace: Dict[str, Any],
    repair_evaluation: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Offline evaluation diagnosis.

    This function may use injected/gold metadata because it is evaluation-only.
    None of these fields influence primitive repair decisions.
    """
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
        "missing_required_supporting_passage_ids": sorted(missing_required_support_ids),
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


def finalize_repair_trace(
    original_trace: Dict[str, Any],
    repaired_trace: Dict[str, Any],
    repair_order: Sequence[str],
    repair_mode: str,
    experiment_label: str,
) -> Dict[str, Any]:
    original = deepcopy(original_trace)
    repaired = deepcopy(repaired_trace)
    failures = list(original.get("true_failures", []))

    evidence_changed = _evidence_signature(original) != _evidence_signature(repaired)
    generation: Dict[str, Any] | None = None

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
    else:
        # A true no-op repair must not receive accidental credit or regression
        # from a fresh generation. Preserve the already-observed failure-stage
        # answer and evaluation exactly.
        final_answer = str(
            original.get("failure_answer")
            or original.get("final_answer")
            or original.get("baseline_answer")
            or ""
        )
        repair_evaluation = deepcopy(
            original.get("failure_evaluation") or original.get("evaluation", {})
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
    diagnosis["final_generation_skipped"] = not evidence_changed

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

    # M/D structural regressions are measurable from passage membership. L is
    # evaluated separately through the injected L target in offline diagnosis.
    structural_regression = any(failure in {"M", "D"} for failure in new_failures)
    regression_detected = bool(semantic_regression or structural_regression)

    diagnosis["semantic_regression_detected"] = semantic_regression
    diagnosis["structural_regression_detected"] = structural_regression

    repaired["experiment_stage"] = (
        "single_repair" if len(failures) == 1 else f"compound_repair_{repair_mode}"
    )
    repaired["repair_type"] = "+".join(failures)
    repaired["repair_mode"] = repair_mode
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
    repaired["repair_generation_skipped"] = not evidence_changed

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

    prompt_hashes = dict(repaired.get("prompt_hashes", {}))
    if generation is not None:
        prompt_hashes[f"repair_generation_{repair_mode}"] = generation.get(
            "prompt_hash", ""
        )
        repaired["repair_latency_ms"] = int(generation.get("latency_ms", 0))
        repaired["repair_token_usage"] = generation.get("token_usage", {})
    else:
        repaired["repair_latency_ms"] = 0
        repaired["repair_token_usage"] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    repaired["prompt_hashes"] = prompt_hashes
    return repaired


def process_single_trace(
    source_trace: Dict[str, Any],
    repair_order: Sequence[str],
    repair_mode: str,
    experiment_label: str,
) -> Dict[str, Any]:
    """Helper function to execute a single trace for the ThreadPoolExecutor."""
    original = deepcopy(source_trace)
    repaired = deepcopy(original)
    failures = set(original.get("true_failures", []))

    for failure in repair_order:
        if failure in failures:
            repaired = apply_repair(repaired, failure)

    repaired = finalize_repair_trace(
        original,
        repaired,
        repair_order,
        repair_mode,
        experiment_label,
    )
    return repaired


def run_repair(
    input_path: str,
    output_path: str,
    repair_order: Sequence[str],
    repair_mode: str,
    experiment_label: str,
) -> None:
    with open(input_path, "r", encoding="utf-8") as file:
        traces = json.load(file)

    results = []
    timeouts = []

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=1)

    try:
        for index, source_trace in enumerate(traces, start=1):

            job = pool.apply_async(
                process_single_trace,
                (
                    source_trace,
                    repair_order,
                    repair_mode,
                    experiment_label,
                ),
            )

            try:
                repaired = job.get(timeout=TRACE_TIMEOUT_SECONDS)
            except mp.TimeoutError:
                timeout_trace = deepcopy(source_trace)
                timeout_trace["runtime"] = {
                    "timed_out": True,
                    "stage": f"repair_{experiment_label}",
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
            except Exception as e:
                print(
                    f"{experiment_label} skipped trace {index}/{len(traces)}: ERROR ({e})"
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

    print(f"Saved repair results: {output_path}")
    print(f"Saved repair timeouts: {timeout_path} ({len(timeouts)})")


def main():
    for label, input_path, output_path, order, mode in EXPERIMENTS:
        run_repair(input_path, output_path, order, mode, label)


if __name__ == "__main__":
    mp.freeze_support()
    main()
