import json

REQUIRED_FIELDS = [
    "trace_id",
    "question",
    "canonical_answer",
    "retrieval_events",
    "true_failures",
    "final_answer",
    "evaluation",
]


VALID_FAILURES = {"M", "D", "G"}


def validate_trace(trace):

    errors = []

    # Required fields
    for field in REQUIRED_FIELDS:
        if field not in trace:
            errors.append(f"Missing field: {field}")

    # Failure labels
    if "true_failures" in trace:

        for failure in trace["true_failures"]:

            if failure not in VALID_FAILURES:
                errors.append(f"Invalid failure type: {failure}")

    # Retrieval check
    if "retrieval_events" in trace:

        if not isinstance(trace["retrieval_events"], list):
            errors.append("retrieval_events must be a list")

    # Evaluation check
    if "evaluation" in trace:

        required_eval = ["exact_match", "semantic_correct"]

        for item in required_eval:

            if item not in trace["evaluation"]:
                errors.append(f"Missing evaluation field: {item}")

    return errors


def validate_file(path):

    with open(path, "r", encoding="utf-8") as f:

        traces = json.load(f)

    total_errors = 0

    for index, trace in enumerate(traces):

        errors = validate_trace(trace)

        if errors:

            print(f"\nTrace {index} failed:")

            for error in errors:
                print(" -", error)

            total_errors += len(errors)

    if total_errors == 0:
        print("Validation passed. All traces are valid.")
    else:
        print(f"Validation failed. Total errors: {total_errors}")


if __name__ == "__main__":

    validate_file("data/results/single/missing_evidence_results.json")
