import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42

EXPERIMENTS = {
    "baseline": "data/processed/pilot_baseline_results.json",
    "M": "data/results/single/missing_evidence_results.json",
    "D": "data/results/single/distractor_results.json",
    "L": "data/results/single/linkage_results.json",
    "M_D": "data/results/compound/M_D_results.json",
    "M_L": "data/results/compound/M_L_results.json",
    "D_L": "data/results/compound/D_L_results.json",
    "M_D_L": "data/results/compound/M_D_L_results.json",
}

SKIPPED = {
    "M": "data/failures/single/skipped/missing_evidence_skipped.json",
    "D": "data/failures/single/skipped/distractor_skipped.json",
    "L": "data/failures/single/skipped/linkage_skipped.json",
    "M_D": "data/failures/compound/skipped/M_D_skipped.json",
    "M_L": "data/failures/compound/skipped/M_L_skipped.json",
    "D_L": "data/failures/compound/skipped/D_L_skipped.json",
    "M_D_L": "data/failures/compound/skipped/M_D_L_skipped.json",
}

PAIR_COMPOUNDS = {
    "M_D": ("M", "D"),
    "M_L": ("M", "L"),
    "D_L": ("D", "L"),
}

# These comparisons expose cancellation effects, such as an added D making an
# answer correct again after L had made it wrong.
CROSS_CONDITION_PAIRS = {
    # Add the second failure to each single condition.
    "M_to_M_D": ("M", "M_D"),
    "D_to_M_D": ("D", "M_D"),
    "M_to_M_L": ("M", "M_L"),
    "L_to_M_L": ("L", "M_L"),
    "D_to_D_L": ("D", "D_L"),
    "L_to_D_L": ("L", "D_L"),
    # Add the third failure to each pair condition.
    "M_D_to_M_D_L": ("M_D", "M_D_L"),
    "M_L_to_M_D_L": ("M_L", "M_D_L"),
    "D_L_to_M_D_L": ("D_L", "M_D_L"),
}

PRECONDITION_PREFIXES = (
    "baseline_generation_failed",
    "baseline_semantic_judge_error",
    "baseline_incorrect",
    "incomplete_supporting_evidence",
    "fewer_than_two_retrieved_supporting_passages",
    "no_non_supporting_passage_to_replace",
    "not_exactly_two",
    "l_requires_exactly_two",
)

MODEL_FAILURE_TOKENS = (
    "call_failed",
    "structured_output",
    "output_unavailable",
    "parser",
    "parse_error",
)

INJECTION_RUNTIME_ERROR_TOKENS = (
    "failure_injection_error:",
    "injection_error:",
)

INVALID_INJECTION_TOKENS = (
    "invalid_",
    "empty_",
    "unchanged",
    "mismatch",
    "source_and_target_are_same",
    "not_found",
    "still_visible",
    "artifact_not_applicable",
    "injection_error",
    "compound_artifact_not_applicable",
)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return value


def _evaluation(trace: Dict[str, Any], stage: str) -> Dict[str, Any]:
    if stage == "baseline":
        return trace.get("baseline_evaluation") or trace.get("evaluation", {})
    return trace.get("failure_evaluation") or trace.get("evaluation", {})


def _technical_error(evaluation: Dict[str, Any]) -> bool:
    return bool(
        evaluation.get("generation_failed", False)
        or evaluation.get("semantic_judge_error", False)
    )


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round(fraction * (len(ordered) - 1)))),
    )
    return ordered[index]


def bootstrap_ci(values: List[float]) -> List[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(sample) / len(sample))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def calculate_metrics(traces: List[Dict[str, Any]], stage: str) -> Dict[str, Any]:
    """Calculate explicit all-trace and evaluable-only metrics.

    Primary accuracy fields use only evaluable traces. Technical failures are
    reported separately instead of silently counting as ordinary wrong answers.
    """
    evaluations = [_evaluation(trace, stage) for trace in traces]
    total = len(evaluations)
    evaluable = [item for item in evaluations if not _technical_error(item)]
    evaluable_count = len(evaluable)

    all_em = [float(bool(item.get("exact_match", False))) for item in evaluations]
    all_f1 = [float(item.get("token_f1", 0.0)) for item in evaluations]
    all_semantic = [
        float(bool(item.get("semantic_correct", False))) for item in evaluations
    ]

    em = [float(bool(item.get("exact_match", False))) for item in evaluable]
    f1 = [float(item.get("token_f1", 0.0)) for item in evaluable]
    semantic = [float(bool(item.get("semantic_correct", False))) for item in evaluable]

    return {
        # Backward-compatible count. For a failure condition this means accepted
        # result traces, not all attempted source traces.
        "total": total,
        "accepted_count": total,
        "evaluable_count": evaluable_count,
        "exact_match_count": int(sum(em)),
        "semantic_correct_count": int(sum(semantic)),
        "exact_match_accuracy": sum(em) / evaluable_count if evaluable_count else 0.0,
        "mean_token_f1": sum(f1) / evaluable_count if evaluable_count else 0.0,
        "semantic_accuracy": (
            sum(semantic) / evaluable_count if evaluable_count else 0.0
        ),
        "exact_match_ci95": bootstrap_ci(em),
        "token_f1_ci95": bootstrap_ci(f1),
        "semantic_ci95": bootstrap_ci(semantic),
        "all_trace_exact_match_accuracy": sum(all_em) / total if total else 0.0,
        "all_trace_mean_token_f1": sum(all_f1) / total if total else 0.0,
        "all_trace_semantic_accuracy": (sum(all_semantic) / total if total else 0.0),
        "generation_failure_count": sum(
            bool(item.get("generation_failed", False)) for item in evaluations
        ),
        "semantic_judge_error_count": sum(
            bool(item.get("semantic_judge_error", False)) for item in evaluations
        ),
    }


def _trace_map(traces: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        trace_id: trace
        for trace in traces
        if (trace_id := str(trace.get("trace_id", "")))
    }


def _traces_for_ids(
    mapping: Dict[str, Dict[str, Any]], trace_ids: Iterable[str]
) -> List[Dict[str, Any]]:
    """Return one trace per ID in deterministic order.

    This avoids duplicate rows and guarantees that every paired calculation
    uses exactly the same IDs on both sides.
    """
    return [
        mapping[trace_id] for trace_id in sorted(set(trace_ids)) if trace_id in mapping
    ]


def _trace_id_quality(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    ids = [str(trace.get("trace_id", "")) for trace in traces]
    nonempty = [trace_id for trace_id in ids if trace_id]
    return {
        "missing_trace_id_count": len(ids) - len(nonempty),
        "duplicate_trace_count": len(nonempty) - len(set(nonempty)),
    }


def transitions(
    baseline: List[Dict[str, Any]],
    failure: List[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_map = _trace_map(baseline)
    failure_map = _trace_map(failure)
    common_ids = sorted(set(baseline_map) & set(failure_map))
    counts = Counter()
    effectiveness: List[float] = []

    for trace_id in common_ids:
        before = _evaluation(baseline_map[trace_id], "baseline")
        after = _evaluation(failure_map[trace_id], "failure")
        counts["matched_traces"] += 1
        if _technical_error(before) or _technical_error(after):
            counts["indeterminate_traces"] += 1
            continue

        counts["evaluable_traces"] += 1
        before_correct = bool(before.get("semantic_correct", False))
        after_correct = bool(after.get("semantic_correct", False))
        if before_correct:
            counts["baseline_correct_count"] += 1

        if before_correct and not after_correct:
            counts["correct_to_wrong"] += 1
        elif not before_correct and after_correct:
            counts["wrong_to_correct"] += 1
        elif before_correct and after_correct:
            counts["stayed_correct"] += 1
        else:
            counts["stayed_wrong"] += 1

        if before_correct:
            effectiveness.append(float(not after_correct))

    return {
        "baseline_correct_count": counts["baseline_correct_count"],
        "correct_to_wrong": counts["correct_to_wrong"],
        "wrong_to_correct": counts["wrong_to_correct"],
        "stayed_correct": counts["stayed_correct"],
        "stayed_wrong": counts["stayed_wrong"],
        "matched_traces": counts["matched_traces"],
        "evaluable_traces": counts["evaluable_traces"],
        "indeterminate_traces": counts["indeterminate_traces"],
        "failure_effectiveness": (
            sum(effectiveness) / len(effectiveness) if effectiveness else 0.0
        ),
        "failure_effectiveness_ci95": bootstrap_ci(effectiveness),
    }


def _strip_component_prefix(reason: str) -> str:
    value = reason.strip()
    while ":" in value:
        prefix, rest = value.split(":", 1)
        if prefix.strip() in {"M", "D", "L", "M_D", "M_L", "D_L", "M_D_L"}:
            value = rest.strip()
        else:
            break
    return value


def _skip_category(reason: str) -> str:
    normalized = _strip_component_prefix(reason).casefold()
    if normalized.startswith(PRECONDITION_PREFIXES):
        return "precondition"
    # Exceptions raised while constructing an injection (for example a TypeError)
    # are runtime/call failures, not evidence artifacts that merely failed validation.
    if any(token in normalized for token in INJECTION_RUNTIME_ERROR_TOKENS):
        return "injection_call_failure"
    if any(token in normalized for token in MODEL_FAILURE_TOKENS):
        return "injection_call_failure"
    if any(token in normalized for token in INVALID_INJECTION_TOKENS):
        return "invalid_injection"
    return "other"


def skip_summary(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        return {
            "skipped_count": 0,
            "precondition_skip_count": 0,
            "injection_call_failure_count": 0,
            "invalid_injection_skip_count": 0,
            "other_skip_count": 0,
            "skip_reason_counts": {},
        }

    items = load_json(path)
    reasons = Counter(str(item.get("reason", "unknown")) for item in items)
    categories = Counter(
        _skip_category(str(item.get("reason", "unknown"))) for item in items
    )
    return {
        "skipped_count": len(items),
        "precondition_skip_count": categories["precondition"],
        "injection_call_failure_count": categories["injection_call_failure"],
        "invalid_injection_skip_count": categories["invalid_injection"],
        "other_skip_count": categories["other"],
        "skip_reason_counts": dict(reasons),
    }


def _history(trace: Dict[str, Any], failure: str) -> Dict[str, Any]:
    return next(
        (
            item
            for item in trace.get("failure_history", [])
            if item.get("failure") == failure
        ),
        {},
    )


def _component_valid(trace: Dict[str, Any], failure: str) -> bool:
    documents = trace.get("retrieval_events", [])
    ids = {str(item.get("passage_id", "")) for item in documents}
    docs_by_id = {
        str(item.get("passage_id", "")): item
        for item in documents
        if item.get("passage_id")
    }
    text_by_id = {
        passage_id: str(item.get("text", ""))
        for passage_id, item in docs_by_id.items()
    }
    gold_ids = {
        str(item)
        for item in trace.get("gold_supporting_passage_ids", [])
        if item
    }

    entry = _history(trace, failure)
    if not entry:
        return False

    if failure == "M":
        removed = {
            str(item) for item in entry.get("removed_passage_ids", []) if item
        }
        if not removed or not removed.isdisjoint(ids):
            return False
        # M is defined as removing supporting evidence. When explicit gold IDs
        # are available, verify that the removed artifact came from that set.
        if gold_ids and not removed.issubset(gold_ids):
            return False
        return True

    if failure == "D":
        source_id = str(entry.get("source_passage_id", ""))
        replaced_id = str(entry.get("replaced_passage_id", ""))
        distractor_id = str(entry.get("distractor_passage_id", ""))
        distractor = docs_by_id.get(distractor_id)
        original_claim = str(entry.get("original_claim", "")).strip()
        modified_claim = str(entry.get("modified_claim", "")).strip()

        if not source_id or not replaced_id or not distractor_id or distractor is None:
            return False
        if source_id == replaced_id:
            return False
        if replaced_id in ids:
            return False
        if gold_ids and source_id not in gold_ids:
            return False
        if not str(distractor.get("text", "")).strip():
            return False
        if str(distractor.get("document_role", "")) != "distractor":
            return False
        if str(distractor.get("source_passage_id", "")) != source_id:
            return False
        if str(distractor.get("replaced_passage_id", "")) != replaced_id:
            return False
        if not original_claim or not modified_claim or original_claim == modified_claim:
            return False
        return True

    if failure == "L":
        target_id = str(entry.get("target_passage_id", ""))
        source_ids = {
            str(item) for item in entry.get("source_passage_ids", []) if item
        }
        link_type = str(entry.get("link_type", "")).strip()
        original = str(entry.get("original_passage_text", "")).strip()
        corrupted = str(entry.get("corrupted_passage_text", "")).strip()

        # L-v16 has two valid structural forms:
        #   1) passage_to_passage: one gold source passage links to a different
        #      gold target passage.
        #   2) question_to_passage: the question itself supplies the identity,
        #      so source_passage_ids is intentionally empty.
        #
        # The old analyzer required source_ids for every L trace, which
        # incorrectly marked every question_to_passage injection as invalid.
        if link_type == "passage_to_passage":
            if not source_ids or target_id in source_ids:
                return False
        elif link_type == "question_to_passage":
            if source_ids:
                return False
        else:
            # Current final L-v16 always records the link type. Reject unknown
            # forms rather than silently treating malformed metadata as valid.
            return False

        if not (
            target_id
            and target_id in ids
            and original
            and corrupted
            and original != corrupted
            and text_by_id.get(target_id, "").strip() == corrupted
        ):
            return False

        # The edited target must be a gold support. For passage_to_passage, the
        # source identity must also refer to gold support metadata. The source
        # passage does not have to remain in retrieval_events because a later M
        # intervention is allowed to remove it while preserving the L artifact.
        if gold_ids:
            if target_id not in gold_ids:
                return False
            if link_type == "passage_to_passage" and not source_ids.issubset(gold_ids):
                return False

        return True

    return False


def _target_separation_valid(trace: Dict[str, Any], failures: List[str]) -> bool:
    """Validate only collision rules that the frozen compound generator enforces.

    M_L and M_D_L explicitly protect the L-corrupted target from M removal.
    D_L, however, is constructed by preserving the validated D distractor and
    then applying L to supporting evidence. The frozen generator does not
    require D's source_passage_id to differ from L's target_passage_id, so the
    analyzer must not invent that additional restriction.
    """
    if "L" not in failures:
        return True

    l_target = str(_history(trace, "L").get("target_passage_id", ""))
    if not l_target:
        return False

    if "M" in failures:
        removed_by_m = {
            str(item)
            for item in _history(trace, "M").get("removed_passage_ids", [])
            if item
        }
        if l_target in removed_by_m:
            return False

    return True


def injection_validity(name: str, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures = name.split("_")
    expected_failures = set(failures)
    valid = sum(
        set(str(item) for item in trace.get("true_failures", [])) == expected_failures
        and all(_component_valid(trace, failure) for failure in failures)
        and _target_separation_valid(trace, failures)
        for trace in traces
    )
    return {
        "valid_injection_count": valid,
        "invalid_saved_injection_count": len(traces) - valid,
        "injection_validity_rate": valid / len(traces) if traces else 0.0,
    }


def _metric_drop(
    baseline: List[Dict[str, Any]],
    condition: List[Dict[str, Any]],
) -> Dict[str, float]:
    base = calculate_metrics(baseline, "baseline")
    after = calculate_metrics(condition, "failure")
    return {
        "em": base["exact_match_accuracy"] - after["exact_match_accuracy"],
        "token_f1": base["mean_token_f1"] - after["mean_token_f1"],
        "semantic": base["semantic_accuracy"] - after["semantic_accuracy"],
    }


def pair_interaction(
    name: str,
    results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    first, second = PAIR_COMPOUNDS[name]
    maps = {key: _trace_map(results[key]) for key in ("baseline", first, second, name)}
    common = set.intersection(*(set(mapping) for mapping in maps.values()))
    evaluable = {
        trace_id
        for trace_id in common
        if not any(
            _technical_error(
                _evaluation(
                    maps[key][trace_id],
                    "baseline" if key == "baseline" else "failure",
                )
            )
            for key in maps
        )
    }
    if not evaluable:
        return {
            "common_trace_count": len(common),
            "evaluable_common_trace_count": 0,
        }

    baseline = _traces_for_ids(maps["baseline"], evaluable)
    first_traces = _traces_for_ids(maps[first], evaluable)
    second_traces = _traces_for_ids(maps[second], evaluable)
    compound = _traces_for_ids(maps[name], evaluable)
    first_drop = _metric_drop(baseline, first_traces)
    second_drop = _metric_drop(baseline, second_traces)
    compound_drop = _metric_drop(baseline, compound)
    return {
        "common_trace_count": len(common),
        "evaluable_common_trace_count": len(evaluable),
        "single_drops": {first: first_drop, second: second_drop},
        "compound_drop": compound_drop,
        "extra_semantic_drop_vs_strongest_single": compound_drop["semantic"]
        - max(first_drop["semantic"], second_drop["semantic"]),
    }


def triple_interaction(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    names = ["baseline", "M", "D", "L", "M_D", "M_L", "D_L", "M_D_L"]
    maps = {name: _trace_map(results[name]) for name in names}
    common = set.intersection(*(set(mapping) for mapping in maps.values()))
    evaluable = {
        trace_id
        for trace_id in common
        if not any(
            _technical_error(
                _evaluation(
                    maps[name][trace_id],
                    "baseline" if name == "baseline" else "failure",
                )
            )
            for name in names
        )
    }
    if not evaluable:
        return {
            "common_trace_count": len(common),
            "evaluable_common_trace_count": 0,
        }

    baseline = _traces_for_ids(maps["baseline"], evaluable)
    drops = {
        name: _metric_drop(
            baseline,
            _traces_for_ids(maps[name], evaluable),
        )
        for name in names[1:]
    }
    strongest_pair = max(
        ("M_D", "M_L", "D_L"),
        key=lambda pair_name: drops[pair_name]["semantic"],
    )
    return {
        "common_trace_count": len(common),
        "evaluable_common_trace_count": len(evaluable),
        "single_drops": {name: drops[name] for name in ("M", "D", "L")},
        "pair_drops": {name: drops[name] for name in ("M_D", "M_L", "D_L")},
        "triple_drop": drops["M_D_L"],
        "strongest_pair": strongest_pair,
        "extra_semantic_drop_vs_strongest_pair": drops["M_D_L"]["semantic"]
        - drops[strongest_pair]["semantic"],
    }


def _cross_condition_transition(
    source_name: str,
    target_name: str,
    results: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    source_map = _trace_map(results[source_name])
    target_map = _trace_map(results[target_name])
    baseline_map = _trace_map(results["baseline"])
    common_ids = sorted(set(source_map) & set(target_map) & set(baseline_map))
    counts = Counter()
    rows: List[Dict[str, Any]] = []

    for trace_id in common_ids:
        baseline_eval = _evaluation(baseline_map[trace_id], "baseline")
        source_eval = _evaluation(source_map[trace_id], "failure")
        target_eval = _evaluation(target_map[trace_id], "failure")

        if any(
            _technical_error(item) for item in (baseline_eval, source_eval, target_eval)
        ):
            transition = "indeterminate"
            counts[transition] += 1
        else:
            source_correct = bool(source_eval.get("semantic_correct", False))
            target_correct = bool(target_eval.get("semantic_correct", False))
            if source_correct and not target_correct:
                transition = "correct_to_wrong"
            elif not source_correct and target_correct:
                transition = "wrong_to_correct"
            elif source_correct and target_correct:
                transition = "stayed_correct"
            else:
                transition = "stayed_wrong"
            counts[transition] += 1

        rows.append(
            {
                "comparison": f"{source_name}_to_{target_name}",
                "trace_id": trace_id,
                "baseline_correct": bool(baseline_eval.get("semantic_correct", False)),
                "source_condition": source_name,
                "source_correct": bool(source_eval.get("semantic_correct", False)),
                "target_condition": target_name,
                "target_correct": bool(target_eval.get("semantic_correct", False)),
                "transition": transition,
            }
        )

    summary = {
        "common_trace_count": len(common_ids),
        "correct_to_wrong": counts["correct_to_wrong"],
        "wrong_to_correct": counts["wrong_to_correct"],
        "stayed_correct": counts["stayed_correct"],
        "stayed_wrong": counts["stayed_wrong"],
        "indeterminate": counts["indeterminate"],
    }
    return summary, rows


def build_cross_condition_transitions(
    results: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summaries: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for label, (source_name, target_name) in CROSS_CONDITION_PAIRS.items():
        summary, comparison_rows = _cross_condition_transition(
            source_name,
            target_name,
            results,
        )
        summaries[label] = summary
        rows.extend(comparison_rows)
    return summaries, rows


def save_transition_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fields = [
        "comparison",
        "trace_id",
        "baseline_correct",
        "source_condition",
        "source_correct",
        "target_condition",
        "target_correct",
        "transition",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_csv(summary: Dict[str, Any], path: Path) -> None:
    fields = [
        "condition",
        "attempted_count",
        "accepted_count",
        "skipped_count",
        "precondition_skip_count",
        "injection_call_failure_count",
        "invalid_injection_skip_count",
        "other_skip_count",
        "evaluable_count",
        "matched_baseline_count",
        "matched_evaluable_count",
        "matched_baseline_exact_match_accuracy",
        "matched_baseline_mean_token_f1",
        "matched_baseline_semantic_accuracy",
        "exact_match_accuracy",
        "mean_token_f1",
        "semantic_accuracy",
        "em_drop",
        "token_f1_drop",
        "semantic_drop",
        "baseline_correct_count",
        "correct_to_wrong",
        "wrong_to_correct",
        "stayed_correct",
        "stayed_wrong",
        "failure_effectiveness",
        "valid_injection_count",
        "invalid_saved_injection_count",
        "invalid_injection_count",
        "generation_failure_count",
        "semantic_judge_error_count",
        "unmatched_trace_count",
    ]

    rows = []
    for name, item in summary.items():
        transition = item.get("transitions", {})
        matched_baseline = item.get("matched_baseline_evaluable", {})
        rows.append(
            {
                "condition": name,
                "attempted_count": item.get("attempted_count", item.get("total", 0)),
                "accepted_count": item.get("accepted_count", item.get("total", 0)),
                "skipped_count": item.get("skipped_count", 0),
                "precondition_skip_count": item.get("precondition_skip_count", 0),
                "injection_call_failure_count": item.get(
                    "injection_call_failure_count", 0
                ),
                "invalid_injection_skip_count": item.get(
                    "invalid_injection_skip_count", 0
                ),
                "other_skip_count": item.get("other_skip_count", 0),
                "evaluable_count": item.get("evaluable_count", 0),
                "matched_baseline_count": item.get("matched_baseline_count", ""),
                "matched_evaluable_count": item.get("matched_evaluable_count", ""),
                "matched_baseline_exact_match_accuracy": matched_baseline.get(
                    "exact_match_accuracy", ""
                ),
                "matched_baseline_mean_token_f1": matched_baseline.get(
                    "mean_token_f1", ""
                ),
                "matched_baseline_semantic_accuracy": matched_baseline.get(
                    "semantic_accuracy", ""
                ),
                "exact_match_accuracy": item.get("exact_match_accuracy", 0.0),
                "mean_token_f1": item.get("mean_token_f1", 0.0),
                "semantic_accuracy": item.get("semantic_accuracy", 0.0),
                "em_drop": item.get("em_drop", ""),
                "token_f1_drop": item.get("token_f1_drop", ""),
                "semantic_drop": item.get("semantic_drop", ""),
                "baseline_correct_count": transition.get("baseline_correct_count", ""),
                "correct_to_wrong": transition.get("correct_to_wrong", ""),
                "wrong_to_correct": transition.get("wrong_to_correct", ""),
                "stayed_correct": transition.get("stayed_correct", ""),
                "stayed_wrong": transition.get("stayed_wrong", ""),
                "failure_effectiveness": transition.get("failure_effectiveness", ""),
                "valid_injection_count": item.get("valid_injection_count", ""),
                "invalid_saved_injection_count": item.get(
                    "invalid_saved_injection_count", ""
                ),
                "invalid_injection_count": item.get("invalid_injection_count", ""),
                "generation_failure_count": item.get("generation_failure_count", 0),
                "semantic_judge_error_count": item.get("semantic_judge_error_count", 0),
                "unmatched_trace_count": item.get("unmatched_trace_count", ""),
            }
        )

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    results = {name: load_json(path) for name, path in EXPERIMENTS.items()}
    baseline = results["baseline"]
    baseline_map = _trace_map(baseline)

    baseline_metrics = calculate_metrics(baseline, "baseline")
    baseline_metrics.update(_trace_id_quality(baseline))
    baseline_metrics.update(
        {
            "attempted_count": len(baseline),
            "accepted_count": len(baseline),
            "skipped_count": 0,
            "precondition_skip_count": 0,
            "injection_call_failure_count": 0,
            "invalid_injection_skip_count": 0,
            "other_skip_count": 0,
        }
    )
    summary: Dict[str, Any] = {"baseline": baseline_metrics}

    for name in ("M", "D", "L", "M_D", "M_L", "D_L", "M_D_L"):
        traces = results[name]
        condition_map = _trace_map(traces)
        matched_ids = set(condition_map) & set(baseline_map)
        evaluable_ids = {
            trace_id
            for trace_id in matched_ids
            if not _technical_error(_evaluation(baseline_map[trace_id], "baseline"))
            and not _technical_error(_evaluation(condition_map[trace_id], "failure"))
        }

        matched_baseline = _traces_for_ids(baseline_map, matched_ids)
        evaluable_baseline = _traces_for_ids(baseline_map, evaluable_ids)
        evaluable_condition = _traces_for_ids(condition_map, evaluable_ids)

        metrics = calculate_metrics(traces, "failure")
        skip_metrics = skip_summary(SKIPPED[name])
        validity = injection_validity(name, traces)
        matched_baseline_metrics = calculate_metrics(matched_baseline, "baseline")
        matched_baseline_evaluable = calculate_metrics(evaluable_baseline, "baseline")
        matched_condition_evaluable = calculate_metrics(evaluable_condition, "failure")

        invalid_injection_count = (
            validity["invalid_saved_injection_count"]
            + skip_metrics["invalid_injection_skip_count"]
        )

        metrics.update(
            {
                # The old eligible_count field is retained as an alias for
                # compatibility, but accepted_count is the accurate name.
                "eligible_count": len(traces),
                "accepted_count": len(traces),
                "attempted_count": len(traces) + skip_metrics["skipped_count"],
                **_trace_id_quality(traces),
                **skip_metrics,
                "unmatched_trace_count": len(condition_map) - len(matched_ids),
                "matched_baseline_count": len(matched_ids),
                "matched_evaluable_count": len(evaluable_ids),
                "matched_baseline": matched_baseline_metrics,
                "matched_baseline_evaluable": matched_baseline_evaluable,
                "matched_condition_evaluable": matched_condition_evaluable,
                "em_drop": matched_baseline_evaluable["exact_match_accuracy"]
                - matched_condition_evaluable["exact_match_accuracy"],
                "token_f1_drop": matched_baseline_evaluable["mean_token_f1"]
                - matched_condition_evaluable["mean_token_f1"],
                "semantic_drop": matched_baseline_evaluable["semantic_accuracy"]
                - matched_condition_evaluable["semantic_accuracy"],
                "transitions": transitions(baseline, traces),
                **validity,
                "invalid_injection_count": invalid_injection_count,
            }
        )

        if name in PAIR_COMPOUNDS:
            metrics["compound_interaction"] = pair_interaction(name, results)
        if name == "M_D_L":
            metrics["triple_interaction"] = triple_interaction(results)
        summary[name] = metrics

    cross_summary, transition_rows = build_cross_condition_transitions(results)
    summary["cross_condition_transitions"] = cross_summary

    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "failure_summary.json"
    csv_path = output_dir / "failure_summary.csv"
    transition_csv_path = output_dir / "condition_transitions.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    save_csv(
        {name: value for name, value in summary.items() if name in EXPERIMENTS},
        csv_path,
    )
    save_transition_csv(transition_rows, transition_csv_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {transition_csv_path}")


if __name__ == "__main__":
    main()