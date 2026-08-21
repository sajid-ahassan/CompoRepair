import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from .pipeline.baseline_rag import (
    evaluate_prediction,
    generate_answer,
    model_manifest,
)
from .repair.distractor import repair_distractor
from .repair.missing_evidence import repair_missing_evidence
from .repair.reasoning import repair_reasoning

EXPERIMENTS = [
    (
        "data/results/single/missing_evidence_results.json",
        "data/repaired_result/M_repair_results.json",
    ),
    (
        "data/results/single/distractor_results.json",
        "data/repaired_result/D_repair_results.json",
    ),
    (
        "data/results/single/reasoning_results.json",
        "data/repaired_result/G_repair_results.json",
    ),
    (
        "data/results/compound/M_D_results.json",
        "data/repaired_result/M_D_repair_results.json",
    ),
    (
        "data/results/compound/M_G_results.json",
        "data/repaired_result/M_G_repair_results.json",
    ),
    (
        "data/results/compound/D_G_results.json",
        "data/repaired_result/D_G_repair_results.json",
    ),
]


def _supporting_ids(trace: Dict[str, Any]) -> set:
    return {
        str(item.get("passage_id", ""))
        for item in trace.get("selected_evidence", [])
        if item.get("is_supporting", False)
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


def _update_g_repair_history(
    repaired_trace: Dict[str, Any],
    old_answer: str,
    generation: Dict[str, Any],
) -> None:
    """Attach the shared normal-generation result to the G repair record."""
    g_entry = next(
        (
            item
            for item in reversed(repaired_trace.get("repair_history", []))
            if item.get("failure") == "G"
        ),
        None,
    )
    if g_entry is None:
        return

    generation_failed = bool(generation.get("generation_failed", False))
    final_answer = str(generation.get("text", "") or "")
    dependency_restored = bool(g_entry.get("dependency_restored", False))

    g_entry.update(
        {
            "answer_kept": final_answer == str(old_answer or ""),
            "answer_regenerated": not generation_failed,
            "repair_applied": dependency_restored and not generation_failed,
            "generation_failed": generation_failed,
            "generation_mode": "normal",
            "generation_pending": False,
            "generation_prompt_hash": generation.get("prompt_hash", ""),
            "generation_latency_ms": int(generation.get("latency_ms", 0)),
            "generation_token_usage": generation.get("token_usage", {}),
        }
    )


def diagnose_failures_after_repair(
    original_trace: Dict[str, Any],
    repaired_trace: Dict[str, Any],
    repair_evaluation: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    final_ids = {
        str(document.get("passage_id", ""))
        for document in repaired_trace.get("retrieval_events", [])
    }

    # M is present whenever any required gold-supporting passage is absent
    # after repair. Restoration of the specifically injected M passage is
    # recorded separately so controlled repair success remains auditable.
    injected_missing_ids = _injected_missing_ids(original_trace)
    remaining_injected_missing_ids = injected_missing_ids - final_ids

    required_support = _supporting_ids(original_trace)
    missing_required_support_ids = required_support - final_ids
    support_complete_after = not missing_required_support_ids
    m_after = bool(missing_required_support_ids)

    # D remains only when the specifically injected distractor is present.
    known_distractors = _injected_distractor_ids(original_trace)
    remaining_injected_distractors = known_distractors & final_ids
    d_after = bool(remaining_injected_distractors)

    generation_failed = bool(
        repair_evaluation.get("generation_failed", False)
    )
    semantic_judge_error = bool(
        repair_evaluation.get("semantic_judge_error", False)
    )
    semantic_correct = bool(
        repair_evaluation.get("semantic_correct", False)
    )

    # G remains when the answer is wrong despite complete supporting
    # evidence and removal of the injected distractor.
    g_after = bool(
        not generation_failed
        and not semantic_judge_error
        and not semantic_correct
        and not m_after
        and not d_after
    )

    failures = []

    if m_after:
        failures.append("M")

    if d_after:
        failures.append("D")

    if g_after:
        failures.append("G")

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
        "remaining_injected_distractor_ids": sorted(
            remaining_injected_distractors
        ),
        "M_present": m_after,
        "D_present": d_after,
        "G_present": g_after,
        "M_diagnosis_basis": "gold_support_coverage",
        "support_complete_after": support_complete_after,
        "D_diagnosis_basis": "injected_distractor_presence",
        "generation_failed": generation_failed,
        "semantic_judge_error": semantic_judge_error,
    }

    return failures, diagnosis


def run_repair(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as file:
        traces = json.load(file)

    results = []
    for index, source_trace in enumerate(traces, start=1):
        original = deepcopy(source_trace)
        repaired = deepcopy(original)
        failures = list(original.get("true_failures", []))
        old_answer = str(
            original.get("failure_answer")
            or original.get("final_answer", "")
            or ""
        )

        # Fixed repair order: M -> D -> G.
        if "M" in failures:
            repaired = repair_missing_evidence(repaired)
        if "D" in failures:
            repaired = repair_distractor(repaired)
        if "G" in failures:
            repaired = repair_reasoning(repaired)

        # Every repaired condition uses the same normal answer generator.
        generation = generate_answer(
            repaired["question"],
            repaired.get("retrieval_events", []),
        )
        final_answer = str(generation.get("text", "") or "")
        repaired["final_answer"] = final_answer

        if "G" in failures:
            _update_g_repair_history(repaired, old_answer, generation)

        repair_evaluation = evaluate_prediction(
            repaired["question"],
            final_answer,
            repaired["canonical_answer"],
            generation_failed=bool(generation.get("generation_failed", False)),
        )

        failure_after, diagnosis = diagnose_failures_after_repair(
            original,
            repaired,
            repair_evaluation,
        )

        # A failed generation cannot be counted as a successful full repair.
        # Conservatively keep the originally targeted failures unresolved.
        outcome_indeterminate = bool(
            repair_evaluation.get("generation_failed", False)
            or repair_evaluation.get("semantic_judge_error", False)
        )

        if outcome_indeterminate:
            # Do not count an unevaluable trace as fully repaired.
            failure_after = list(dict.fromkeys([*failure_after, *failures]))

        diagnosis["outcome_indeterminate"] = outcome_indeterminate

        original_failures = set(failures)
        newly_diagnosed_failures = [
            failure
            for failure in failure_after
            if failure not in original_failures
        ]

        before_evaluation = original.get("failure_evaluation") or original.get(
            "evaluation", {}
        )
        before_semantic_evaluable = bool(
            not before_evaluation.get("generation_failed", False)
            and not before_evaluation.get("semantic_judge_error", False)
        )
        after_semantic_evaluable = bool(
            not repair_evaluation.get("semantic_judge_error", False)
        )
        semantic_regression = bool(
            before_semantic_evaluable
            and after_semantic_evaluable
            and before_evaluation.get("semantic_correct", False)
            and not repair_evaluation.get("semantic_correct", False)
        )

        # New M/D labels are observable evidence defects introduced by repair.
        # A newly diagnosed G is repair-induced only when the answer actually
        # changes from semantically correct to incorrect. If it was already
        # wrong before repair, G is residual/latent rather than a regression.
        new_failures = [
            failure
            for failure in newly_diagnosed_failures
            if failure in {"M", "D"}
            or (failure == "G" and semantic_regression)
        ]
        residual_newly_diagnosed = [
            failure
            for failure in newly_diagnosed_failures
            if failure not in new_failures
        ]
        structural_regression = any(
            failure in {"M", "D"} for failure in new_failures
        )
        regression_detected = bool(
            semantic_regression or structural_regression
        )

        diagnosis["semantic_regression_detected"] = semantic_regression
        diagnosis["structural_regression_detected"] = structural_regression
        diagnosis["residual_newly_diagnosed_failures"] = (
            residual_newly_diagnosed
        )

        repaired["experiment_stage"] = (
            "compound_repair" if len(failures) > 1 else "single_repair"
        )
        repaired["repair_type"] = "+".join(failures)
        repaired["repair_evaluation"] = repair_evaluation
        repaired["evaluation"] = repair_evaluation
        repaired["failure_after_repair"] = failure_after
        repaired["new_failures_after_repair"] = new_failures
        repaired["residual_newly_diagnosed_failures"] = (
            residual_newly_diagnosed
        )
        repaired["semantic_regression_detected"] = semantic_regression
        repaired["structural_regression_detected"] = structural_regression
        repaired["regression_detected"] = regression_detected
        repaired["repair_diagnosis"] = diagnosis
        repaired["model_manifest"] = model_manifest()

        repaired["answer_claims"] = (
            [
                {
                    "claim": final_answer,
                    "source": f"repair_{repaired['repair_type']}",
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
        prompt_hashes["repair_generation"] = generation.get("prompt_hash", "")
        repaired["prompt_hashes"] = prompt_hashes
        repaired["repair_latency_ms"] = int(generation.get("latency_ms", 0))
        repaired["repair_token_usage"] = generation.get("token_usage", {})

        results.append(repaired)
        print(f"Repair processed trace {index}/{len(traces)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"Saved repair results: {output_path}")


def main():
    for input_path, output_path in EXPERIMENTS:
        run_repair(input_path, output_path)


if __name__ == "__main__":
    main()
