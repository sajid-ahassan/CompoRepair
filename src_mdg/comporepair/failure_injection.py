import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from .pipeline.baseline_rag import (
    generate_reasoning_plan,
    invoke_structured,
    llm,
)

DISTRACTOR_RANK = 2
G_INTERVENTION_VERSION = "g_dependency_removal_v1"


class DistractorCandidate(BaseModel):
    distractor_text: str
    original_claim: str
    modified_claim: str


def _baseline_eligible(trace: Dict[str, Any]) -> Tuple[bool, str]:
    evaluation = trace.get("baseline_evaluation") or trace.get("evaluation", {})
    if evaluation.get("generation_failed"):
        return False, "baseline_generation_failed"
    if evaluation.get("semantic_judge_error"):
        return False, "baseline_semantic_judge_error"
    if not evaluation.get("semantic_correct"):
        return False, "baseline_incorrect"
    return True, ""


def _split_documents(trace: Dict[str, Any]):
    support_ids = {
        str(item.get("passage_id", ""))
        for item in trace.get("selected_evidence", [])
        if item.get("is_supporting", False)
    }
    documents = trace.get("retrieval_events", [])
    supporting = [
        document
        for document in documents
        if str(document.get("passage_id", "")) in support_ids
    ]
    non_supporting = [
        document
        for document in documents
        if str(document.get("passage_id", "")) not in support_ids
    ]
    present_ids = {str(document.get("passage_id", "")) for document in documents}
    all_support_present = bool(support_ids) and support_ids.issubset(present_ids)
    return supporting, non_supporting, all_support_present


def failure_eligibility(trace: Dict[str, Any], failure: str) -> Tuple[bool, str]:
    eligible, reason = _baseline_eligible(trace)
    if not eligible:
        return False, reason

    supporting, non_supporting, all_support_present = _split_documents(trace)
    if not all_support_present:
        return False, "incomplete_supporting_evidence"

    if failure == "M" and len(supporting) < 2:
        return False, "fewer_than_two_retrieved_supporting_passages"
    if failure == "D" and not non_supporting:
        return False, "no_non_supporting_passage_to_replace"
    if failure == "G" and len(supporting) < 2:
        return False, "fewer_than_two_retrieved_supporting_passages"

    return True, ""


def _rerank(documents: List[Dict[str, Any]]) -> None:
    for rank, document in enumerate(documents, start=1):
        document["rank"] = rank


def _finalize(trace: Dict[str, Any], failures: List[str]) -> Dict[str, Any]:
    trace["true_failures"] = failures
    trace["experiment_stage"] = (
        f"single_failure_{failures[0]}"
        if len(failures) == 1
        else f"compound_failure_{'_'.join(failures)}"
    )
    trace["failure_answer"] = ""
    trace["final_answer"] = trace.get("baseline_answer", "")
    trace.pop("failure_evaluation", None)
    trace["failure_after_repair"] = []
    trace["new_failures_after_repair"] = []
    trace["regression_detected"] = False
    trace["repair_history"] = []
    return trace


def inject_missing_evidence(trace: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(trace)
    supporting, _, _ = _split_documents(result)
    supporting.sort(key=lambda item: int(item.get("rank", 999999)))
    removed = supporting[0]
    removed_id = str(removed["passage_id"])

    result["retrieval_events"] = [
        document
        for document in result["retrieval_events"]
        if str(document.get("passage_id", "")) != removed_id
    ]
    _rerank(result["retrieval_events"])
    result.setdefault("failure_history", []).append(
        {
            "failure": "M",
            "removed_passage_ids": [removed_id],
            "supporting_count_before": len(supporting),
            "supporting_count_after": len(supporting) - 1,
        }
    )
    return result


def _format_documents(documents: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"Title: {document.get('title', '')}\nText: {document.get('text', '')}"
        for document in documents
    )


def _clean_distractor_text(text: str) -> str:
    """Remove accidental title/text wrappers from generated distractors."""
    text = text.strip()

    lower = text.lower()

    if lower.startswith("title:") and "\ntext:" in lower:
        text = text.split("\nText:", 1)[1].strip()
    elif lower.startswith("text:"):
        text = text[5:].strip()

    return text


def inject_distractor(trace: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(trace)
    supporting, non_supporting, _ = _split_documents(result)
    supporting.sort(key=lambda item: int(item.get("rank", 999999)))
    non_supporting.sort(key=lambda item: int(item.get("rank", 999999)))

    # Use the last supporting passage so the saved D artifact remains
    # compatible with M+D, where M removes the first supporting passage.
    source = supporting[-1]
    replacement = non_supporting[-1]
    other_support = [item for item in supporting if item is not source]

    prompt = f"""Create one contradictory version of the source passage.

        Question:
        {result["question"]}

        Source title:
        {source.get("title", "")}

        Source text:
        {source.get("text", "")}

        Other supporting passages:
        {_format_documents(other_support)}

        Change exactly one answer-relevant fact in the Source text. Keep the same main entity, relation, topic, title, and writing style. Preserve everything else.

        The other supporting passages are context only. Do not copy their text, entities, or claims into distractor_text. The distractor must remain a modified version of the Source text.

        Return:
        - distractor_text: only the modified Source passage body
        - original_claim
        - modified_claim

        Do not include "Title:", "Text:", headings, labels, or explanations inside distractor_text.
    """

    generator = llm.with_structured_output(
        DistractorCandidate,
        method="json_schema",
    )
    candidate = invoke_structured(generator, prompt)

    source_id = str(source["passage_id"])
    replacement_id = str(replacement["passage_id"])
    distractor_id = f"{replacement_id}__distractor"

    documents = [
        document
        for document in result["retrieval_events"]
        if str(document.get("passage_id", "")) != replacement_id
    ]
    documents.insert(
        min(DISTRACTOR_RANK - 1, len(documents)),
        {
            "title": source.get("title", ""),
            "passage_id": distractor_id,
            "rank": DISTRACTOR_RANK,
            "text": _clean_distractor_text(candidate.distractor_text),
            "is_supporting": False,
            "document_role": "distractor",
            "source_passage_id": source_id,
            "replaced_passage_id": replacement_id,
        },
    )
    _rerank(documents)
    result["retrieval_events"] = documents
    result.setdefault("failure_history", []).append(
        {
            "failure": "D",
            "source_passage_id": source_id,
            "replaced_passage_id": replacement_id,
            "distractor_passage_id": distractor_id,
            "original_claim": candidate.original_claim,
            "modified_claim": candidate.modified_claim,
            "distractor_rank": DISTRACTOR_RANK,
        }
    )
    return result


def inject_reasoning_failure(trace: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(trace)

    clean_plan = generate_reasoning_plan(
        result["question"],
        result["retrieval_events"],
    )
    corrupted_plan = deepcopy(clean_plan)
    final_step = corrupted_plan["steps"][-1]
    removed_dependency = final_step["depends_on"].pop()

    result["reasoning_plan"] = clean_plan
    result["corrupted_reasoning_plan"] = corrupted_plan
    result["g_removed_dependency"] = removed_dependency
    result["g_corrupted_step_id"] = final_step["id"]

    result.setdefault("failure_history", []).append(
        {
            "failure": "G",
            "intervention_version": G_INTERVENTION_VERSION,
            "failure_type": "reasoning_dependency_removed",
            "corrupted_step_id": final_step["id"],
            "removed_dependency": removed_dependency,
            "evidence_modified": False,
        }
    )
    return result


def inject_single(
    trace: Dict[str, Any], failure: str
) -> Tuple[Optional[Dict[str, Any]], str]:
    eligible, reason = failure_eligibility(trace, failure)
    if not eligible:
        return None, reason

    if failure == "M":
        return _finalize(inject_missing_evidence(trace), ["M"]), ""
    if failure == "D":
        return _finalize(inject_distractor(trace), ["D"]), ""
    return _finalize(inject_reasoning_failure(trace), ["G"]), ""


def _add_saved_d(m_trace: Dict[str, Any], d_trace: Dict[str, Any]):
    result = deepcopy(m_trace)
    d_history = next(
        item for item in d_trace["failure_history"] if item["failure"] == "D"
    )
    source_id = str(d_history["source_passage_id"])
    replaced_id = str(d_history["replaced_passage_id"])
    current_ids = {
        str(item.get("passage_id", "")) for item in result["retrieval_events"]
    }

    if source_id not in current_ids or replaced_id not in current_ids:
        return None

    distractor = next(
        deepcopy(item)
        for item in d_trace["retrieval_events"]
        if item.get("passage_id") == d_history["distractor_passage_id"]
    )
    documents = [
        item
        for item in result["retrieval_events"]
        if item.get("passage_id") != replaced_id
    ]
    documents.insert(min(DISTRACTOR_RANK - 1, len(documents)), distractor)
    _rerank(documents)
    result["retrieval_events"] = documents
    result.setdefault("failure_history", []).append(deepcopy(d_history))
    return result


def _add_saved_g(
    evidence_trace: Dict[str, Any],
    g_trace: Dict[str, Any],
) -> Dict[str, Any]:
    result = deepcopy(evidence_trace)
    g_history = next(
        item for item in g_trace["failure_history"] if item["failure"] == "G"
    )

    result["reasoning_plan"] = deepcopy(g_trace["reasoning_plan"])
    result["corrupted_reasoning_plan"] = deepcopy(
        g_trace["corrupted_reasoning_plan"]
    )
    result["g_removed_dependency"] = g_trace["g_removed_dependency"]
    result["g_corrupted_step_id"] = g_trace["g_corrupted_step_id"]
    result.setdefault("failure_history", []).append(deepcopy(g_history))
    return result


def _save(
    path: str, traces: List[Dict[str, Any]], skipped: List[Dict[str, Any]]
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(traces, file, indent=2, ensure_ascii=False)
    skipped_path = Path(path).with_name(Path(path).stem + "_skipped.json")
    with open(skipped_path, "w", encoding="utf-8") as file:
        json.dump(skipped, file, indent=2, ensure_ascii=False)
    print(f"Saved {path}: {len(traces)} eligible, {len(skipped)} skipped")


def main():
    baseline_path = "data/processed/pilot_baseline_results.json"
    with open(baseline_path, "r", encoding="utf-8") as file:
        baseline = json.load(file)

    single_paths = {
        "M": "data/failures/single/missing_evidence.json",
        "D": "data/failures/single/distractor.json",
        "G": "data/failures/single/reasoning.json",
    }
    singles = {"M": {}, "D": {}, "G": {}}
    skip_reasons = {"M": {}, "D": {}, "G": {}}

    for failure in ["M", "D", "G"]:
        accepted, skipped = [], []
        for trace in baseline:
            trace_id = str(trace.get("trace_id", ""))
            try:
                result, reason = inject_single(trace, failure)
            except Exception as error:
                result = None
                reason = (
                    f"failure_injection_error:{type(error).__name__}"
                )
            if result is None:
                skip_reasons[failure][trace_id] = reason
                skipped.append(
                    {"trace_id": trace_id, "failures": [failure], "reason": reason}
                )
            else:
                singles[failure][trace_id] = result
                accepted.append(result)
        _save(single_paths[failure], accepted, skipped)

    compound_specs = [
        (["M", "D"], "data/failures/compound/M_D.json"),
        (["M", "G"], "data/failures/compound/M_G.json"),
        (["D", "G"], "data/failures/compound/D_G.json"),
    ]

    for failures, output_path in compound_specs:
        accepted, skipped = [], []
        first, second = failures
        for trace in baseline:
            trace_id = str(trace.get("trace_id", ""))
            if trace_id not in singles[first]:
                reason = f"{first}:{skip_reasons[first].get(trace_id, 'single_failure_ineligible')}"
            elif trace_id not in singles[second]:
                reason = f"{second}:{skip_reasons[second].get(trace_id, 'single_failure_ineligible')}"
            else:
                reason = ""

            if reason:
                skipped.append(
                    {"trace_id": trace_id, "failures": failures, "reason": reason}
                )
                continue

            if failures == ["M", "D"]:
                result = _add_saved_d(singles["M"][trace_id], singles["D"][trace_id])
                if result is None:
                    skipped.append(
                        {
                            "trace_id": trace_id,
                            "failures": failures,
                            "reason": "D:artifact_not_compatible_with_M",
                        }
                    )
                    continue
            else:
                result = _add_saved_g(
                    singles[first][trace_id],
                    singles["G"][trace_id],
                )

            accepted.append(_finalize(result, failures))

        _save(output_path, accepted, skipped)


if __name__ == "__main__":
    main()