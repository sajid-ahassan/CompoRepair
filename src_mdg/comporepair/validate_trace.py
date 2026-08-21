import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .retrieval.vector_store import RETRIEVAL_K

VALID_FAILURES = {"M", "D", "G"}
REQUIRED_IDENTITY_FIELDS = {
    "trace_id",
    "question_id",
    "question",
    "canonical_answer",
    "experiment_stage",
}
REQUIRED_EVALUATION_FIELDS = {
    "exact_match",
    "token_f1",
    "semantic_correct",
    "generation_failed",
    "semantic_judge_error",
}


def _expected_retrieval_k(trace: Dict[str, Any]) -> int:
    value = trace.get("model_manifest", {}).get("retrieval_k")
    try:
        return int(value)
    except (TypeError, ValueError):
        return RETRIEVAL_K


def _evaluation_for_stage(trace: Dict[str, Any]) -> Dict[str, Any]:
    stage = str(trace.get("experiment_stage", ""))
    if stage == "baseline":
        return trace.get("baseline_evaluation", {})
    if "repair" in stage:
        return trace.get("repair_evaluation", {})
    if "failure_result" in stage:
        return trace.get("failure_evaluation", {})
    return {}


def _validate_reasoning_plan(
    plan: Any,
    label: str,
    require_multi_dependency_final: bool,
) -> List[str]:
    errors = []

    if not isinstance(plan, dict):
        return [f"{label} must be a dictionary"]

    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return [f"{label}.steps must be a non-empty list"]

    if len(steps) < 3:
        errors.append(f"{label} must contain at least two evidence steps and one final step")

    step_ids = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"{label}.steps[{index}] must be a dictionary")
            continue

        step_id = str(step.get("id", "")).strip()
        task = str(step.get("task", "")).strip()
        dependencies = step.get("depends_on")

        if not step_id:
            errors.append(f"{label}.steps[{index}] has an empty id")
        step_ids.append(step_id)

        if not task:
            errors.append(f"{label}.steps[{index}] has an empty task")

        if not isinstance(dependencies, list):
            errors.append(f"{label}.steps[{index}].depends_on must be a list")
            continue

        if len(dependencies) != len(set(dependencies)):
            errors.append(f"{label}.steps[{index}].depends_on contains duplicates")

        earlier_ids = set(step_ids[:-1])
        for dependency in dependencies:
            if dependency not in earlier_ids:
                errors.append(
                    f"{label}.steps[{index}] dependency {dependency!r} "
                    "must reference an earlier step"
                )

    non_empty_ids = [step_id for step_id in step_ids if step_id]
    if len(non_empty_ids) != len(set(non_empty_ids)):
        errors.append(f"{label} contains duplicate step IDs")

    final_step = steps[-1] if isinstance(steps[-1], dict) else {}
    final_dependencies = final_step.get("depends_on", [])
    if (
        require_multi_dependency_final
        and isinstance(final_dependencies, list)
        and len(final_dependencies) < 2
    ):
        errors.append(f"{label} final step must depend on at least two earlier steps")

    return errors


def _validate_g_intervention(trace: Dict[str, Any]) -> List[str]:
    errors = []

    clean_plan = trace.get("reasoning_plan")
    corrupted_plan = trace.get("corrupted_reasoning_plan")

    errors.extend(
        _validate_reasoning_plan(
            clean_plan,
            "reasoning_plan",
            require_multi_dependency_final=True,
        )
    )
    errors.extend(
        _validate_reasoning_plan(
            corrupted_plan,
            "corrupted_reasoning_plan",
            require_multi_dependency_final=False,
        )
    )

    if not isinstance(clean_plan, dict) or not isinstance(corrupted_plan, dict):
        return errors

    clean_steps = clean_plan.get("steps", [])
    corrupted_steps = corrupted_plan.get("steps", [])
    if not isinstance(clean_steps, list) or not isinstance(corrupted_steps, list):
        return errors
    if not clean_steps or not corrupted_steps:
        return errors

    if len(clean_steps) != len(corrupted_steps):
        errors.append("G corruption must not add or remove reasoning steps")
        return errors

    for index, (clean_step, corrupted_step) in enumerate(
        zip(clean_steps, corrupted_steps)
    ):
        if not isinstance(clean_step, dict) or not isinstance(corrupted_step, dict):
            continue

        if clean_step.get("id") != corrupted_step.get("id"):
            errors.append(f"G corruption changed step ID at index {index}")
        if clean_step.get("task") != corrupted_step.get("task"):
            errors.append(f"G corruption changed step task at index {index}")

        if index < len(clean_steps) - 1:
            if clean_step.get("depends_on", []) != corrupted_step.get(
                "depends_on", []
            ):
                errors.append(
                    f"G corruption changed a non-final dependency at step index {index}"
                )

    clean_final = clean_steps[-1]
    corrupted_final = corrupted_steps[-1]
    if not isinstance(clean_final, dict) or not isinstance(corrupted_final, dict):
        return errors

    clean_dependencies = clean_final.get("depends_on", [])
    corrupted_dependencies = corrupted_final.get("depends_on", [])
    if not isinstance(clean_dependencies, list) or not isinstance(
        corrupted_dependencies, list
    ):
        return errors

    removed_dependencies = [
        dependency
        for dependency in clean_dependencies
        if dependency not in corrupted_dependencies
    ]
    added_dependencies = [
        dependency
        for dependency in corrupted_dependencies
        if dependency not in clean_dependencies
    ]

    if len(removed_dependencies) != 1:
        errors.append("G corruption must remove exactly one final-step dependency")
    if added_dependencies:
        errors.append("G corruption must not add final-step dependencies")

    if len(corrupted_dependencies) != len(clean_dependencies) - 1:
        errors.append("G corrupted final step has an invalid dependency count")

    expected_after_removal = [
        dependency
        for dependency in clean_dependencies
        if dependency not in removed_dependencies
    ]
    if corrupted_dependencies != expected_after_removal:
        errors.append(
            "G corruption must preserve dependency order and remove only one dependency"
        )

    recorded_dependency = str(trace.get("g_removed_dependency", ""))
    recorded_step_id = str(trace.get("g_corrupted_step_id", ""))

    if len(removed_dependencies) == 1:
        if recorded_dependency != str(removed_dependencies[0]):
            errors.append("g_removed_dependency does not match the removed dependency")

    if recorded_step_id != str(clean_final.get("id", "")):
        errors.append("g_corrupted_step_id does not match the final step ID")

    failure_history = trace.get("failure_history", [])
    g_entry = next(
        (
            item
            for item in failure_history
            if isinstance(item, dict) and item.get("failure") == "G"
        ),
        {},
    )

    if g_entry.get("evidence_modified") is not False:
        errors.append("G injection must record evidence_modified=False")
    if g_entry.get("failure_type") != "reasoning_dependency_removed":
        errors.append(
            "G failure history must record failure_type=reasoning_dependency_removed"
        )
    if g_entry.get("intervention_version") != "g_dependency_removal_v1":
        errors.append(
            "G failure history must record intervention_version=g_dependency_removal_v1"
        )
    if str(g_entry.get("removed_dependency", "")) != recorded_dependency:
        errors.append(
            "G failure-history removed_dependency does not match g_removed_dependency"
        )
    if str(g_entry.get("corrupted_step_id", "")) != recorded_step_id:
        errors.append(
            "G failure-history corrupted_step_id does not match g_corrupted_step_id"
        )

    return errors


def validate_trace(trace: Dict[str, Any]) -> List[str]:
    errors = []
    missing = REQUIRED_IDENTITY_FIELDS - set(trace)
    errors.extend(f"Missing field: {field}" for field in sorted(missing))

    failures = trace.get("true_failures", [])
    if not isinstance(failures, list):
        errors.append("true_failures must be a list")
        failures = []

    invalid_failures = [
        failure for failure in failures if failure not in VALID_FAILURES
    ]
    if invalid_failures:
        errors.append(f"Invalid failure labels: {invalid_failures}")
    if len(failures) != len(set(failures)):
        errors.append("true_failures contains duplicates")

    retrieval = trace.get("retrieval_events", [])
    if not isinstance(retrieval, list):
        errors.append("retrieval_events must be a list")
        retrieval = []

    passage_ids = [str(item.get("passage_id", "")) for item in retrieval]
    if any(not passage_id for passage_id in passage_ids):
        errors.append("retrieval_events contains an empty passage_id")
    if len(passage_ids) != len(set(passage_ids)):
        errors.append("retrieval_events contains duplicate passage IDs")

    stage = str(trace.get("experiment_stage", ""))
    is_repair_stage = "repair" in stage
    is_failure_stage = "failure" in stage and not is_repair_stage

    if "G" in failures:
        errors.extend(_validate_g_intervention(trace))
    else:
        if trace.get("reasoning_plan"):
            errors.append("reasoning_plan is present without a G failure")
        if trace.get("corrupted_reasoning_plan"):
            errors.append("corrupted_reasoning_plan is present without a G failure")
        if trace.get("reasoning_outputs"):
            errors.append("reasoning_outputs is present without a G failure")

    expected_retrieval_k = _expected_retrieval_k(trace)

    if stage == "baseline" and len(retrieval) != expected_retrieval_k:
        errors.append(
            f"baseline retrieval count must be {expected_retrieval_k}, "
            f"found {len(retrieval)}"
        )

    if is_failure_stage:
        expected_count = expected_retrieval_k - (1 if "M" in failures else 0)
        if len(retrieval) != expected_count:
            errors.append(
                f"failure retrieval count must be {expected_count}, "
                f"found {len(retrieval)}"
            )

    if is_repair_stage and len(retrieval) > expected_retrieval_k:
        errors.append(
            "repaired retrieval count cannot exceed "
            f"{expected_retrieval_k}, found {len(retrieval)}"
        )

    failure_history = trace.get("failure_history", [])
    history_labels = [
        item.get("failure")
        for item in failure_history
        if isinstance(item, dict)
    ]
    for failure in failures:
        if history_labels.count(failure) != 1:
            errors.append(
                f"failure_history must contain exactly one {failure} entry"
            )

    if "M" in failures and is_failure_stage:
        entry = next(
            (
                item
                for item in failure_history
                if isinstance(item, dict) and item.get("failure") == "M"
            ),
            {},
        )
        if any(
            passage_id in passage_ids
            for passage_id in entry.get("removed_passage_ids", [])
        ):
            errors.append("M injection did not remove all recorded passage IDs")

    if "D" in failures and is_failure_stage:
        entry = next(
            (
                item
                for item in failure_history
                if isinstance(item, dict) and item.get("failure") == "D"
            ),
            {},
        )
        distractor_id = entry.get("distractor_passage_id")
        source_id = entry.get("source_passage_id")

        if distractor_id not in passage_ids:
            errors.append("D injection distractor is not present")
        if source_id not in passage_ids:
            errors.append("D injection source passage is not retained")

    if "G" in failures and ("failure_result" in stage or is_repair_stage):
        control_evaluation = trace.get("g_clean_control_evaluation", {})
        if not isinstance(control_evaluation, dict) or not control_evaluation:
            errors.append("G result is missing g_clean_control_evaluation")
        else:
            missing_control_fields = REQUIRED_EVALUATION_FIELDS - set(
                control_evaluation
            )
            errors.extend(
                f"Missing G clean-control evaluation field: {field}"
                for field in sorted(missing_control_fields)
            )

            control_answer = str(
                trace.get("g_clean_control_answer", "") or ""
            ).strip()
            if (
                not control_answer
                and not control_evaluation.get("generation_failed", False)
            ):
                errors.append(
                    "g_clean_control_answer is empty but its generation_failed "
                    "flag is not true"
                )

        if trace.get("g_clean_control_reused_reasoning_outputs") is not True:
            errors.append(
                "G clean control must reuse the corrupted run's fixed "
                "reasoning outputs"
            )

    evaluation = _evaluation_for_stage(trace)
    if evaluation:
        missing_evaluation = REQUIRED_EVALUATION_FIELDS - set(evaluation)
        errors.extend(
            f"Missing evaluation field: {field}"
            for field in sorted(missing_evaluation)
        )

        answer_field = "final_answer"
        if stage == "baseline":
            answer_field = "baseline_answer"
        elif "failure_result" in stage:
            answer_field = "failure_answer"

        answer = trace.get(answer_field, "")
        if not str(answer).strip() and not evaluation.get(
            "generation_failed", False
        ):
            errors.append(
                f"{answer_field} is empty but generation_failed is not true"
            )

    if is_repair_stage:
        repair_labels = [
            item.get("failure")
            for item in trace.get("repair_history", [])
            if isinstance(item, dict)
        ]
        for failure in failures:
            if repair_labels.count(failure) != 1:
                errors.append(
                    f"repair_history must contain exactly one {failure} entry"
                )

        remaining = trace.get("failure_after_repair", [])
        new_failures = trace.get("new_failures_after_repair", [])

        if any(failure not in remaining for failure in new_failures):
            errors.append(
                "new_failures_after_repair must be a subset of "
                "failure_after_repair"
            )

        semantic_regression = bool(
            trace.get("semantic_regression_detected", False)
        )
        structural_regression = bool(
            trace.get("structural_regression_detected", False)
        )
        regression_detected = bool(
            trace.get("regression_detected", False)
        )

        expected_structural_regression = any(
            failure in {"M", "D"} for failure in new_failures
        )
        if structural_regression != expected_structural_regression:
            errors.append(
                "structural_regression_detected does not match new M/D "
                "failures"
            )

        if "G" in new_failures and not semantic_regression:
            errors.append(
                "a newly introduced G failure requires a semantic "
                "correct-to-wrong transition"
            )

        if regression_detected != bool(
            semantic_regression or structural_regression
        ):
            errors.append(
                "regression_detected does not match semantic or structural "
                "regression"
            )

    return errors


def validate_file(path: str) -> int:
    with open(path, "r", encoding="utf-8") as file:
        traces = json.load(file)

    seen_trace_ids = set()
    total_errors = 0

    for index, trace in enumerate(traces):
        trace_id = str(trace.get("trace_id", ""))
        errors = validate_trace(trace)

        if trace_id in seen_trace_ids:
            errors.append("Duplicate trace_id in file")
        seen_trace_ids.add(trace_id)

        if errors:
            print(f"\nTrace {index} ({trace_id or 'missing-id'}) failed:")
            for error in errors:
                print(f" - {error}")
            total_errors += len(errors)

    if total_errors:
        print(f"Validation failed for {path}. Total errors: {total_errors}")
    else:
        print(f"Validation passed: {path}")

    return total_errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate CompoRepair trace files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSON trace files to validate",
    )
    args = parser.parse_args()

    total_errors = 0
    for path in args.paths:
        if not Path(path).exists():
            print(f"Missing file: {path}")
            total_errors += 1
            continue

        total_errors += validate_file(path)

    raise SystemExit(1 if total_errors else 0)


if __name__ == "__main__":
    main()