from copy import deepcopy
from typing import Any, Dict


def _find_step(plan: Dict[str, Any], step_id: str) -> Dict[str, Any]:
    for step in plan.get("steps", []):
        if str(step.get("id", "")) == step_id:
            return step
    return {}


def repair_reasoning(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the G intervention before normal answer regeneration.

    G failure generation executes ``corrupted_reasoning_plan``. During repair,
    the clean plan is restored only as an auditable repair action. The final
    repaired answer is intentionally generated later by the normal generator
    in ``run_repair.py`` for every failure condition.
    """
    repaired = deepcopy(trace)

    clean_plan = repaired.get("reasoning_plan", {})
    corrupted_plan = repaired.get("corrupted_reasoning_plan", {})
    restored_step_id = str(repaired.get("g_corrupted_step_id", ""))
    restored_dependency = str(repaired.get("g_removed_dependency", ""))

    clean_step = _find_step(clean_plan, restored_step_id)
    corrupted_step = _find_step(corrupted_plan, restored_step_id)
    clean_dependencies = {
        str(item) for item in clean_step.get("depends_on", [])
    }
    corrupted_dependencies = {
        str(item) for item in corrupted_step.get("depends_on", [])
    }

    dependency_restored = bool(
        restored_step_id
        and restored_dependency
        and restored_dependency in clean_dependencies
        and restored_dependency not in corrupted_dependencies
    )

    # Preserve the failure-stage planned outputs for audit, but do not leave
    # them marked as outputs of the repaired normal-generation stage.
    failure_outputs = deepcopy(repaired.get("reasoning_outputs", {}))
    if failure_outputs:
        repaired["failure_reasoning_outputs"] = failure_outputs
    repaired["reasoning_outputs"] = {}

    # Restore the clean plan only as auditable state. It is never passed to
    # the repair-stage answer generator, which receives only question and
    # repaired retrieval evidence.
    repaired["repaired_reasoning_plan"] = deepcopy(clean_plan)
    repaired["g_intervention_active"] = False

    repaired.setdefault("repair_history", []).append(
        {
            "failure": "G",
            "repair": "restore_reasoning_dependency",
            "restored_step_id": restored_step_id,
            "restored_dependency": restored_dependency,
            "dependency_restored": dependency_restored,
            "generation_mode": "normal",
            "generation_pending": True,
        }
    )
    return repaired