from copy import deepcopy

from .distractor import repair_distractor
from .missing_evidence import repair_missing_evidence
from .linkage import repair_linkage


def repair_compound(trace):
    repaired = deepcopy(trace)
    failures = repaired.get("true_failures", [])

    if "M" in failures:
        repaired = repair_missing_evidence(repaired)
    if "D" in failures:
        repaired = repair_distractor(repaired)
    if "L" in failures:
        repaired = repair_linkage(repaired)

    repaired["repair_type"] = "+".join(failures)
    repaired["repair_order"] = [
        failure for failure in ("M", "D", "L") if failure in failures
    ]
    return repaired
