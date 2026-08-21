from copy import deepcopy

from .distractor import repair_distractor
from .missing_evidence import repair_missing_evidence
from .reasoning import repair_reasoning


def repair_compound(trace):
    repaired = deepcopy(trace)
    failures = repaired.get("true_failures", [])

    if "M" in failures:
        repaired = repair_missing_evidence(repaired)
    if "D" in failures:
        repaired = repair_distractor(repaired)
    if "G" in failures:
        repaired = repair_reasoning(repaired)

    repaired["repair_type"] = "+".join(failures)
    return repaired
