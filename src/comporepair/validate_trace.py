import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_RETRIEVAL_K = 8

VALID_FAILURES = {"M", "D", "L"}
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
        return DEFAULT_RETRIEVAL_K


def _evaluation_for_stage(trace: Dict[str, Any]) -> Dict[str, Any]:
    stage = str(trace.get("experiment_stage", ""))
    if stage == "baseline":
        return trace.get("baseline_evaluation", {})
    if "repair" in stage:
        return trace.get("repair_evaluation", {})
    if "failure_result" in stage:
        return trace.get("failure_evaluation", {})
    return {}


def _history_entry(trace: Dict[str, Any], failure: str) -> Dict[str, Any]:
    return next(
        (
            item
            for item in trace.get("failure_history", [])
            if isinstance(item, dict) and item.get("failure") == failure
        ),
        {},
    )


def validate_trace(trace: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    missing = REQUIRED_IDENTITY_FIELDS - set(trace)
    errors.extend(f"Missing field: {field}" for field in sorted(missing))

    failures = trace.get("true_failures", [])
    if not isinstance(failures, list):
        errors.append("true_failures must be a list")
        failures = []
    invalid = [failure for failure in failures if failure not in VALID_FAILURES]
    if invalid:
        errors.append(f"Invalid failure labels: {invalid}")
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
    text_by_id = {
        str(item.get("passage_id", "")): str(item.get("text", ""))
        for item in retrieval
    }

    stage = str(trace.get("experiment_stage", ""))
    is_repair_stage = "repair" in stage
    is_failure_stage = "failure" in stage and not is_repair_stage
    expected_k = _expected_retrieval_k(trace)

    if stage == "baseline" and len(retrieval) != expected_k:
        errors.append(
            f"baseline retrieval count must be {expected_k}, found {len(retrieval)}"
        )
    if is_failure_stage:
        expected_count = expected_k - (1 if "M" in failures else 0)
        if len(retrieval) != expected_count:
            errors.append(
                f"failure retrieval count must be {expected_count}, found {len(retrieval)}"
            )
    if is_repair_stage and len(retrieval) > expected_k:
        errors.append(
            f"repaired retrieval count cannot exceed {expected_k}, found {len(retrieval)}"
        )

    history_labels = [
        item.get("failure")
        for item in trace.get("failure_history", [])
        if isinstance(item, dict)
    ]
    for failure in failures:
        if history_labels.count(failure) != 1:
            errors.append(
                f"failure_history must contain exactly one {failure} entry"
            )

    if "M" in failures and is_failure_stage:
        entry = _history_entry(trace, "M")
        removed = [
            str(item) for item in entry.get("removed_passage_ids", []) if item
        ]
        if not removed:
            errors.append("M failure history has no removed passage ID")
        if any(passage_id in passage_ids for passage_id in removed):
            errors.append("M injection did not remove all recorded passage IDs")

    if "D" in failures and is_failure_stage:
        entry = _history_entry(trace, "D")
        distractor_id = str(entry.get("distractor_passage_id", ""))
        source_id = str(entry.get("source_passage_id", ""))
        m_removed = {
            str(item)
            for item in _history_entry(trace, "M").get("removed_passage_ids", [])
            if item
        }
        if not distractor_id or distractor_id not in passage_ids:
            errors.append("D injection distractor is not present")
        if source_id not in passage_ids and source_id not in m_removed:
            errors.append("D source passage is neither retained nor removed by M")

    if "L" in failures and is_failure_stage:
        entry = _history_entry(trace, "L")
        target_id = str(entry.get("target_passage_id", ""))
        corrupted_text = str(entry.get("corrupted_passage_text", ""))
        if not target_id or target_id not in passage_ids:
            errors.append("L target passage is not present")
        elif text_by_id.get(target_id) != corrupted_text:
            errors.append("L target text does not match the recorded corrupted text")

        if "M" in failures:
            removed_by_m = {
                str(item)
                for item in _history_entry(trace, "M").get(
                    "removed_passage_ids", []
                )
                if item
            }
            if target_id in removed_by_m:
                errors.append("M must not remove the L target passage")

        if "D" in failures:
            d_source_id = str(
                _history_entry(trace, "D").get("source_passage_id", "")
            )
            if target_id and target_id == d_source_id:
                errors.append("D must not use the L target passage as its source")

    evaluation = _evaluation_for_stage(trace)
    if evaluation:
        missing_fields = REQUIRED_EVALUATION_FIELDS - set(evaluation)
        errors.extend(
            f"Missing evaluation field: {field}"
            for field in sorted(missing_fields)
        )
        answer_field = "final_answer"
        if stage == "baseline":
            answer_field = "baseline_answer"
        elif "failure_result" in stage:
            answer_field = "failure_answer"
        if not str(trace.get(answer_field, "")).strip() and not evaluation.get(
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

        order = trace.get("repair_order", [])
        if not isinstance(order, list):
            errors.append("repair_order must be a list")
        elif set(order) != set(failures) or len(order) != len(failures):
            errors.append("repair_order must contain every true failure exactly once")

        remaining = trace.get("failure_after_repair", [])
        new_failures = trace.get("new_failures_after_repair", [])
        if any(failure not in remaining for failure in new_failures):
            errors.append(
                "new_failures_after_repair must be a subset of failure_after_repair"
            )
        if any(failure not in VALID_FAILURES for failure in remaining):
            errors.append("failure_after_repair contains an invalid label")

        semantic_regression = bool(
            trace.get("semantic_regression_detected", False)
        )
        structural_regression = bool(
            trace.get("structural_regression_detected", False)
        )
        regression_detected = bool(trace.get("regression_detected", False))
        expected_structural = any(
            failure in {"M", "D"} for failure in new_failures
        )
        if structural_regression != expected_structural:
            errors.append(
                "structural_regression_detected does not match newly introduced failures"
            )
        if regression_detected != bool(
            semantic_regression or structural_regression
        ):
            errors.append(
                "regression_detected does not match semantic or structural regression"
            )

        if trace.get("repair_mode") == "safe" and not trace.get(
            "safe_composer_history"
        ):
            errors.append("safe repair trace is missing safe_composer_history")

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
    parser.add_argument("paths", nargs="+", help="JSON trace files to validate")
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
