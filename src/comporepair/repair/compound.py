from .missing_evidence import repair_missing_evidence
from .distractor import repair_distractor
from .reasoning import repair_reasoning


def repair_compound(trace):

    repaired_trace = trace.copy()

    failures = trace.get("true_failures", [])

    repair_history = []

    # Order matters:
    # 1. Recover evidence
    # 2. Clean evidence
    # 3. Improve reasoning

    if "M" in failures:

        repaired_trace = repair_missing_evidence(repaired_trace)

        repair_history.extend(repaired_trace.get("repair_history", []))

    if "D" in failures:

        repaired_trace = repair_distractor(repaired_trace)

        repair_history.extend(repaired_trace.get("repair_history", []))

    if "G" in failures:

        repaired_trace = repair_reasoning(repaired_trace)

        repair_history.extend(repaired_trace.get("repair_history", []))

    repaired_trace["repair_history"] = repair_history

    repaired_trace["repair_type"] = "+".join(failures)

    return repaired_trace
