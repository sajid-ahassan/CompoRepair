import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

FINAL_SAMPLE_SIZE = int(os.getenv("COMPOREPAIR_FINAL_SAMPLE_SIZE", "108"))
FINAL_SAMPLE_SEED = int(os.getenv("COMPOREPAIR_FINAL_SAMPLE_SEED", "42"))

CONDITIONS = {
    "M": (
        "data/failures/single/missing_evidence.json",
        "data/final_test_inputs/single/missing_evidence.json",
    ),
    "D": (
        "data/failures/single/distractor.json",
        "data/final_test_inputs/single/distractor.json",
    ),
    "L": (
        "data/failures/single/linkage.json",
        "data/final_test_inputs/single/linkage.json",
    ),
    "M_D": (
        "data/failures/compound/M_D.json",
        "data/final_test_inputs/compound/M_D.json",
    ),
    "M_L": (
        "data/failures/compound/M_L.json",
        "data/final_test_inputs/compound/M_L.json",
    ),
    "D_L": (
        "data/failures/compound/D_L.json",
        "data/final_test_inputs/compound/D_L.json",
    ),
    "M_D_L": (
        "data/failures/compound/M_D_L.json",
        "data/final_test_inputs/compound/M_D_L.json",
    ),
}

MANIFEST_PATH = "data/final_test_inputs/selection_manifest.json"


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return value


def _trace_id(trace: Dict[str, Any]) -> str:
    return str(trace.get("trace_id", "")).strip()


def _select(label: str, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if FINAL_SAMPLE_SIZE <= 0:
        raise ValueError("COMPOREPAIR_FINAL_SAMPLE_SIZE must be greater than zero.")

    if len(traces) < FINAL_SAMPLE_SIZE:
        raise ValueError(
            f"{label}: requested {FINAL_SAMPLE_SIZE} final traces, "
            f"but only {len(traces)} eligible traces are available."
        )

    trace_ids = [_trace_id(trace) for trace in traces]
    if any(not trace_id for trace_id in trace_ids):
        raise ValueError(f"{label}: every eligible trace must have a trace_id.")
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError(f"{label}: duplicate trace_id values found in eligible input.")

    # A condition-specific deterministic RNG makes each selection reproducible
    # without coupling one condition's sample to another condition's list size.
    rng = random.Random(f"{FINAL_SAMPLE_SEED}:{label}")
    selected_indices = rng.sample(range(len(traces)), FINAL_SAMPLE_SIZE)
    return [traces[index] for index in selected_indices]


def main() -> None:
    manifest: Dict[str, Any] = {
        "sample_size": FINAL_SAMPLE_SIZE,
        "sample_seed": FINAL_SAMPLE_SEED,
        "selection_rule": "uniform_random_without_replacement",
        "conditions": {},
    }

    for label, (input_path, output_path) in CONDITIONS.items():
        traces = _load(input_path)
        selected = _select(label, traces)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(selected, file, indent=2, ensure_ascii=False)

        manifest["conditions"][label] = {
            "source_path": input_path,
            "output_path": output_path,
            "eligible_count": len(traces),
            "selected_count": len(selected),
            "selected_trace_ids": [_trace_id(trace) for trace in selected],
        }
        print(f"{label}: selected {len(selected)} of {len(traces)} eligible traces")

    manifest_path = Path(MANIFEST_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(f"Saved final-test selection manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
