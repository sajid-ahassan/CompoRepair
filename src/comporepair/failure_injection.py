import json
import multiprocessing as mp
import os
import re
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from .pipeline.baseline_rag import invoke_structured, llm

DISTRACTOR_RANK = 2
D_INTERVENTION_VERSION = (
    "d_direct_relation_matched_answer_sentence_v7_2_controlled_relaxed_parser"
)
L_INTERVENTION_VERSION = "l_gold_link_entity_anonymization_v17_strict_exact_link"
FAILURE_INJECTION_TIMEOUT_SECONDS = 180.0


class DistractorFakeAnswerCandidate(BaseModel):
    fake_answer: str = Field(
        description=(
            "One short plausible incorrect answer of the requested semantic type. "
            "It must not be equivalent to the gold answer."
        )
    )


class DistractorFakeAnswerProposal(BaseModel):
    can_inject: bool = Field(
        description="True when at least one plausible wrong answer can be proposed."
    )
    candidates: List[DistractorFakeAnswerCandidate] = Field(
        description="Up to five plausible wrong answers, ordered strongest-first."
    )


class DistractorRelationPlanProposal(BaseModel):
    can_map: bool = Field(
        description=(
            "True only when the question can be mapped exactly to one of the "
            "allowed existing distractor relation schemas."
        )
    )
    relation_key: str = Field(
        description="One allowed existing relation key, or empty when can_map=false."
    )
    pair: List[str] = Field(
        default_factory=list,
        description="Exactly two compared entities for pair/binary relations; otherwise empty.",
    )
    subject: str = Field(
        default="",
        description="Question subject for subject-based relations; otherwise empty.",
    )
    work: str = Field(
        default="",
        description="Work/song/film/book for work-based relations; otherwise empty.",
    )
    relative: str = Field(
        default="",
        description="mother/father/spouse/wife/husband for relative_death_date; otherwise empty.",
    )


class LinkBridgeDecision(BaseModel):
    can_inject: bool = Field(
        description=(
            "True when one answer-critical bridge identity can be obscured "
            "without changing the underlying factual evidence."
        )
    )
    source_passage: str = Field(
        description="Local source passage label such as P1, P2, P3, or P4."
    )
    target_passage: str = Field(
        description="Different local target passage label such as P1, P2, P3, or P4."
    )
    bridge_identity: str = Field(
        description="The ONE semantic bridge identity whose explicit linkage should be weakened."
    )
    proposed_bridge_phrases: List[str] = Field(
        description=(
            "One or more phrases that refer to the SAME bridge identity, written "
            "as close as possible to how they appear in the target passage."
        )
    )
    obscured_bridge_reference: str = Field(
        description=(
            "One natural underspecified replacement such as 'the British peer', "
            "'the singer', 'the film', or 'the organization'."
        )
    )
    broken_link_description: str = Field(
        description="One short sentence describing the bridge being obscured."
    )


class ControlledLinkCorruption(BaseModel):
    corrupted_passage_text: str = Field(
        description=(
            "The complete target passage after changing only wording semantically "
            "corresponding to the supplied bridge phrase(s) into the supplied vague "
            "replacement."
        )
    )


def _baseline_eligible(trace: Dict[str, Any]) -> Tuple[bool, str]:
    evaluation = trace.get("baseline_evaluation") or trace.get("evaluation", {})
    if evaluation.get("generation_failed"):
        return False, "baseline_generation_failed"
    if evaluation.get("semantic_judge_error"):
        return False, "baseline_semantic_judge_error"
    if not evaluation.get("semantic_correct"):
        return False, "baseline_incorrect"
    return True, ""


def _gold_support_ids(trace: Dict[str, Any]) -> Set[str]:
    explicit = {
        str(passage_id)
        for passage_id in trace.get("gold_supporting_passage_ids", [])
        if passage_id
    }
    if explicit:
        return explicit
    return {
        str(item.get("passage_id", ""))
        for item in trace.get("selected_evidence", [])
        if item.get("is_supporting", False) and item.get("passage_id")
    }


def _split_documents(trace: Dict[str, Any]):
    support_ids = _gold_support_ids(trace)
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
    if failure == "L" and not (2 <= len(supporting) <= 4):
        return False, "L_requires_two_to_four_retrieved_supporting_passages"
    if failure == "D" and not non_supporting:
        return False, "no_non_supporting_passage_to_replace"

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
    trace["semantic_regression_detected"] = False
    trace["structural_regression_detected"] = False
    trace["regression_detected"] = False
    trace["repair_history"] = []
    trace.pop("repair_diagnosis", None)
    trace.pop("repair_evaluation", None)
    trace.pop("repair_mode", None)
    trace.pop("repair_order", None)
    trace.pop("safe_composer_history", None)
    return trace


def _history_entry(trace: Dict[str, Any], failure: str) -> Dict[str, Any]:
    return next(
        (
            item
            for item in trace.get("failure_history", [])
            if item.get("failure") == failure
        ),
        {},
    )


def inject_missing_evidence(
    trace: Dict[str, Any],
    excluded_support_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    result = deepcopy(trace)
    excluded = {str(item) for item in (excluded_support_ids or [])}
    supporting, _, _ = _split_documents(result)
    supporting.sort(key=lambda item: int(item.get("rank", 999999)))
    candidates = [
        item for item in supporting if str(item.get("passage_id", "")) not in excluded
    ]
    if not candidates:
        raise ValueError("No supporting passage is available for M injection.")

    removed = candidates[0]
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


def _format_passage_texts(documents: Iterable[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"Passage ID: {document.get('passage_id', '')}\n"
        f"Text: {document.get('text', '')}"
        for document in documents
    )


def _clean_distractor_text(text: str) -> str:
    """Remove accidental wrappers from a generated distractor body."""
    text = text.strip()
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("title:"):
        for index, line in enumerate(lines[1:], start=1):
            if line.strip().lower().startswith("text:"):
                first = line.split(":", 1)[1].strip()
                remainder = [first, *lines[index + 1 :]]
                return "\n".join(remainder).strip()
    if text.lower().startswith("text:"):
        return text.split(":", 1)[1].strip()
    return text


def _d_clean_question_text(text: str) -> str:
    text = str(text or "").replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def _d_clean_slot(text: str) -> str:
    return str(text or "").strip().strip(" \t\r\n,.;:?!\"'")


def _d_relation_plan(question: str) -> Optional[Dict[str, Any]]:
    """
    Parse common 2WikiMultiHopQA question forms into a controlled relation schema.

    Only relations with a deterministic answer-sentence template are accepted. Unknown
    forms abstain instead of allowing an LLM to invent a nearby/proxy relation.
    """
    q = _d_clean_question_text(question)
    flags = re.IGNORECASE

    # Pairwise birth/death/age comparisons.
    match = re.match(r"^Was\s+(.+?)\s+or\s+(.+?)\s+born\s+first\?$", q, flags)
    if match:
        return {
            "relation_key": "born_earlier",
            "answer_type": "person",
            "pair": [_d_clean_slot(match.group(1)), _d_clean_slot(match.group(2))],
        }

    match = re.match(
        r"^Who\s+(?:was\s+)?born\s+(first|earlier|later),?\s+(.+?)\s+or\s+(.+?)\?$",
        q,
        flags,
    )
    if match:
        direction = match.group(1).casefold()
        relation_key = "born_later" if direction == "later" else "born_earlier"
        return {
            "relation_key": relation_key,
            "answer_type": "person",
            "pair": [_d_clean_slot(match.group(2)), _d_clean_slot(match.group(3))],
        }

    match = re.match(
        r"^Who\s+died\s+(earlier|later),?\s+(.+?)\s+or\s+(.+?)\?$",
        q,
        flags,
    )
    if match:
        return {
            "relation_key": f"died_{match.group(1).casefold()}",
            "answer_type": "person",
            "pair": [_d_clean_slot(match.group(2)), _d_clean_slot(match.group(3))],
        }

    match = re.match(
        r"^Who\s+(?:is|was)\s+(younger|older),?\s+(.+?)\s+or\s+(.+?)\?$",
        q,
        flags,
    )
    if match:
        return {
            "relation_key": match.group(1).casefold(),
            "answer_type": "person",
            "pair": [_d_clean_slot(match.group(2)), _d_clean_slot(match.group(3))],
        }

    # Film release comparisons.
    match = re.match(
        r"^Which\s+(?:film|movie)\s+was\s+released\s+"
        r"(earlier|later|more\s+recently|most\s+recently),?\s+(.+?)\s+or\s+(.+?)\?$",
        q,
        flags,
    )
    if match:
        direction = re.sub(r"\s+", "_", match.group(1).casefold())
        if direction == "most_recently":
            direction = "more_recently"
        return {
            "relation_key": f"released_{direction}",
            "answer_type": "work",
            "pair": [_d_clean_slot(match.group(2)), _d_clean_slot(match.group(3))],
        }

    # Bridge comparison: compare films through their directors' birth order.
    match = re.match(
        r"^Which\s+(?:film|movie)\s+has\s+the\s+director\s+born\s+"
        r"(earlier|later),?\s+(.+?)\s+or\s+(.+?)\?$",
        q,
        flags,
    )
    if match:
        return {
            "relation_key": f"director_born_{match.group(1).casefold()}",
            "answer_type": "work",
            "pair": [_d_clean_slot(match.group(2)), _d_clean_slot(match.group(3))],
        }

    match = re.match(
        r"^Which\s+(?:film|movie)'s\s+director\s+was\s+born\s+"
        r"(earlier|later),?\s+(.+?)\s+or\s+(.+?)\?$",
        q,
        flags,
    )
    if match:
        return {
            "relation_key": f"director_born_{match.group(1).casefold()}",
            "answer_type": "work",
            "pair": [_d_clean_slot(match.group(2)), _d_clean_slot(match.group(3))],
        }

    # Binary same-country / same-nationality questions.
    binary_patterns = [
        (
            r"^Are\s+the\s+bands\s+(.+?)\s+and\s+(.+?),?\s+from\s+the\s+same\s+country\?$",
            "same_country",
        ),
        (
            r"^(?:Are|Were)\s+(.+?)\s+and\s+(.+?)\s+located\s+in\s+the\s+same\s+country\?$",
            "same_country",
        ),
        (
            r"^(?:Are|Were)\s+(.+?)\s+and\s+(.+?),?\s+from\s+the\s+same\s+country\?$",
            "same_country",
        ),
        (
            r"^(?:Are|Were)\s+(.+?)\s+and\s+(.+?)\s+of\s+the\s+same\s+nationality\?$",
            "same_nationality",
        ),
        (
            r"^Do\s+(.+?)\s+and\s+(.+?)\s+have\s+the\s+same\s+nationality\?$",
            "same_nationality",
        ),
    ]
    for pattern, relation_key in binary_patterns:
        match = re.match(pattern, q, flags)
        if match:
            return {
                "relation_key": relation_key,
                "answer_type": "yes_no",
                "pair": [_d_clean_slot(match.group(1)), _d_clean_slot(match.group(2))],
            }

    # Explicit multi-hop relation questions.
    match = re.match(
        r"^Who\s+(?:is|was)\s+the\s+spouse\s+of\s+the\s+director\s+of\s+"
        r"(?:film\s+)?(.+?)\?$",
        q,
        flags,
    )
    if match:
        work = _d_clean_slot(match.group(1))
        return {
            "relation_key": "spouse_of_director",
            "answer_type": "person",
            "work": work,
            "anchor_hint": work,
        }

    match = re.match(
        r"^Who\s+(?:is|was)\s+(.+?)'s\s+(paternal|maternal)\s+"
        r"(grandfather|grandmother)\?$",
        q,
        flags,
    )
    if match:
        subject = _d_clean_slot(match.group(1))
        side = match.group(2).casefold()
        relation = match.group(3).casefold()
        return {
            "relation_key": f"{side}_{relation}",
            "answer_type": "person",
            "subject": subject,
            "anchor_hint": subject,
        }

    match = re.match(
        r"^Who\s+(?:is|was)\s+(.+?)'s\s+(father|mother)\?$",
        q,
        flags,
    )
    if match:
        subject = _d_clean_slot(match.group(1))
        relation = match.group(2).casefold()
        return {
            "relation_key": relation,
            "answer_type": "person",
            "subject": subject,
            "anchor_hint": subject,
        }

    match = re.match(
        r"^Who\s+(?:is|was)\s+(.+?)'s\s+spouse\?$",
        q,
        flags,
    )
    if match:
        subject = _d_clean_slot(match.group(1))
        return {
            "relation_key": "spouse",
            "answer_type": "person",
            "subject": subject,
            "anchor_hint": subject,
        }

    # Relative's death date; relation is explicit in the generated sentence.
    match = re.match(
        r"^When\s+did\s+(.+?)'s\s+(mother|father|spouse|wife|husband)\s+die\?$",
        q,
        flags,
    )
    if match:
        subject = _d_clean_slot(match.group(1))
        relative = match.group(2).casefold()
        return {
            "relation_key": "relative_death_date",
            "answer_type": "date",
            "subject": subject,
            "relative": relative,
            "anchor_hint": subject,
        }

    match = re.match(r"^When\s+did\s+(.+?)\s+die\?$", q, flags)
    if match:
        subject = _d_clean_slot(match.group(1))
        return {
            "relation_key": "death_date",
            "answer_type": "date",
            "subject": subject,
            "anchor_hint": subject,
        }

    match = re.match(r"^When\s+was\s+(.+?)\s+born\?$", q, flags)
    if match:
        subject = _d_clean_slot(match.group(1))
        return {
            "relation_key": "birth_date",
            "answer_type": "date",
            "subject": subject,
            "anchor_hint": subject,
        }

    # Country / nationality through an explicit performer bridge.
    match = re.match(
        r"^(?:Which|What)\s+country\s+(?:is\s+)?the\s+performer\s+of\s+"
        r"(?:song\s+)?(.+?)\s+(?:is\s+)?from\?$",
        q,
        flags,
    )
    if match:
        work = _d_clean_slot(match.group(1))
        return {
            "relation_key": "performer_country",
            "answer_type": "country",
            "work": work,
            "anchor_hint": work,
        }

    # Direct country / nationality / location questions.
    match = re.match(
        r"^(?:Which|What)\s+country\s+(?:is\s+)?(.+?)\s+(?:is\s+)?from\?$", q, flags
    )
    if match:
        subject = _d_clean_slot(match.group(1))
        return {
            "relation_key": "country",
            "answer_type": "country",
            "subject": subject,
            "anchor_hint": subject,
        }

    match = re.match(r"^Where\s+is\s+(.+?)\s+from\?$", q, flags)
    if match:
        subject = _d_clean_slot(match.group(1))
        return {
            "relation_key": "country",
            "answer_type": "country",
            "subject": subject,
            "anchor_hint": subject,
        }

    match = re.match(r"^(?:What|Which)\s+nationality\s+(?:is|was)\s+(.+?)\?$", q, flags)
    if match:
        subject = _d_clean_slot(match.group(1))
        return {
            "relation_key": "nationality",
            "answer_type": "nationality",
            "subject": subject,
            "anchor_hint": subject,
        }

    match = re.match(r"^Where\s+was\s+(.+?)\s+born\?$", q, flags)
    if match:
        subject = _d_clean_slot(match.group(1))
        return {
            "relation_key": "birth_place",
            "answer_type": "location",
            "subject": subject,
            "anchor_hint": subject,
        }

    # Direct creator relations.
    match = re.match(r"^Who\s+directed\s+(.+?)\?$", q, flags)
    if match:
        work = _d_clean_slot(match.group(1))
        return {
            "relation_key": "director",
            "answer_type": "person",
            "work": work,
            "anchor_hint": work,
        }

    match = re.match(r"^Who\s+(?:is|was)\s+the\s+director\s+of\s+(.+?)\?$", q, flags)
    if match:
        work = _d_clean_slot(match.group(1))
        return {
            "relation_key": "director",
            "answer_type": "person",
            "work": work,
            "anchor_hint": work,
        }

    match = re.match(r"^Who\s+(?:wrote|authored)\s+(.+?)\?$", q, flags)
    if match:
        work = _d_clean_slot(match.group(1))
        return {
            "relation_key": "author",
            "answer_type": "person",
            "work": work,
            "anchor_hint": work,
        }

    match = re.match(r"^Who\s+(?:is|was)\s+the\s+author\s+of\s+(.+?)\?$", q, flags)
    if match:
        work = _d_clean_slot(match.group(1))
        return {
            "relation_key": "author",
            "answer_type": "person",
            "work": work,
            "anchor_hint": work,
        }

    # Direct release date/year.
    match = re.match(r"^What\s+year\s+was\s+(.+?)\s+released\?$", q, flags)
    if match:
        work = _d_clean_slot(match.group(1))
        return {
            "relation_key": "release_year",
            "answer_type": "year",
            "work": work,
            "anchor_hint": work,
        }

    match = re.match(r"^When\s+was\s+(.+?)\s+released\?$", q, flags)
    if match:
        work = _d_clean_slot(match.group(1))
        return {
            "relation_key": "release_date",
            "answer_type": "date",
            "work": work,
            "anchor_hint": work,
        }

    return None


_D_FALLBACK_RELATION_TYPES: Dict[str, str] = {
    "born_earlier": "person",
    "born_later": "person",
    "died_earlier": "person",
    "died_later": "person",
    "younger": "person",
    "older": "person",
    "released_earlier": "work",
    "released_later": "work",
    "released_more_recently": "work",
    "director_born_earlier": "work",
    "director_born_later": "work",
    "same_country": "yes_no",
    "same_nationality": "yes_no",
    "spouse_of_director": "person",
    "paternal_grandfather": "person",
    "maternal_grandfather": "person",
    "paternal_grandmother": "person",
    "maternal_grandmother": "person",
    "father": "person",
    "mother": "person",
    "spouse": "person",
    "relative_death_date": "date",
    "death_date": "date",
    "birth_date": "date",
    "performer_country": "country",
    "country": "country",
    "nationality": "nationality",
    "birth_place": "location",
    "director": "person",
    "author": "person",
    "release_year": "year",
    "release_date": "date",
}


def _d_relation_plan_fallback(question: str) -> Optional[Dict[str, Any]]:
    """
    Controlled fallback for harmless paraphrases that the regex parser misses.

    IMPORTANT:
    - The LLM cannot invent a new relation type.
    - It may map only to a relation already supported by
      _d_render_direct_sentence().
    - Python validates the required fields before accepting the mapping.
    - Unknown/ambiguous questions still abstain.

    This loosens syntax coverage without loosening distractor correctness guards.
    """
    allowed_keys = sorted(_D_FALLBACK_RELATION_TYPES)

    prompt = f"""
Map the question to ONE existing controlled relation schema.

Question:
{question}

Allowed relation keys:
{allowed_keys}

Rules:
- Set can_map=true ONLY when the question's exact requested relation matches
  one allowed relation key.
- Do NOT approximate to a nearby relation.
- Do NOT answer the question.
- Extract entity/work strings exactly from the question as much as possible.
- pair must contain exactly two compared entities for:
  born_earlier, born_later, died_earlier, died_later, younger, older,
  released_earlier, released_later, released_more_recently,
  director_born_earlier, director_born_later, same_country, same_nationality.
- subject is required for:
  father, mother, spouse, paternal_grandfather, maternal_grandfather,
  paternal_grandmother, maternal_grandmother, relative_death_date,
  death_date, birth_date, country, nationality, birth_place.
- work is required for:
  spouse_of_director, performer_country, director, author,
  release_year, release_date.
- relative_death_date also requires relative to be one of:
  mother, father, spouse, wife, husband.
- If the wording is ambiguous or the relation is not in the allowed set,
  set can_map=false.
"""

    runnable = llm.with_structured_output(
        DistractorRelationPlanProposal,
        method="json_schema",
    )

    try:
        proposal = invoke_structured(runnable, prompt)
    except Exception:
        return None

    if not proposal.can_map:
        return None

    relation_key = str(proposal.relation_key or "").strip()
    if relation_key not in _D_FALLBACK_RELATION_TYPES:
        return None

    answer_type = _D_FALLBACK_RELATION_TYPES[relation_key]
    plan: Dict[str, Any] = {
        "relation_key": relation_key,
        "answer_type": answer_type,
    }

    pair_relations = {
        "born_earlier",
        "born_later",
        "died_earlier",
        "died_later",
        "younger",
        "older",
        "released_earlier",
        "released_later",
        "released_more_recently",
        "director_born_earlier",
        "director_born_later",
        "same_country",
        "same_nationality",
    }

    subject_relations = {
        "father",
        "mother",
        "spouse",
        "paternal_grandfather",
        "maternal_grandfather",
        "paternal_grandmother",
        "maternal_grandmother",
        "relative_death_date",
        "death_date",
        "birth_date",
        "country",
        "nationality",
        "birth_place",
    }

    work_relations = {
        "spouse_of_director",
        "performer_country",
        "director",
        "author",
        "release_year",
        "release_date",
    }

    if relation_key in pair_relations:
        pair = [
            _d_clean_slot(item) for item in (proposal.pair or []) if _d_clean_slot(item)
        ]
        if len(pair) != 2 or _d_lexical_norm(pair[0]) == _d_lexical_norm(pair[1]):
            return None
        plan["pair"] = pair
        return plan

    if relation_key in subject_relations:
        subject = _d_clean_slot(proposal.subject)
        if not subject:
            return None
        plan["subject"] = subject
        plan["anchor_hint"] = subject

        if relation_key == "relative_death_date":
            relative = _d_clean_slot(proposal.relative).casefold()
            if relative not in {"mother", "father", "spouse", "wife", "husband"}:
                return None
            plan["relative"] = relative

        return plan

    if relation_key in work_relations:
        work = _d_clean_slot(proposal.work)
        if not work:
            return None
        plan["work"] = work
        plan["anchor_hint"] = work
        return plan

    return None


def _d_lexical_norm(text: str) -> str:
    value = str(text or "").casefold()
    value = (
        value.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
    )
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _d_answer_alias_key(text: str) -> str:
    value = _d_lexical_norm(str(text or ""))
    value = re.sub(r"\b(junior|jr|senior|sr)\b", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _d_country_alias_key(text: str) -> str:
    value = _d_lexical_norm(str(text or ""))
    groups = {
        "united_states": {
            "america",
            "american",
            "united states",
            "united states of america",
            "usa",
            "u s a",
            "us",
            "u s",
        },
        "united_kingdom": {
            "united kingdom",
            "uk",
            "u k",
            "britain",
            "great britain",
            "british",
        },
    }
    for key, aliases in groups.items():
        if value in aliases:
            return key
    return value


def _d_answers_equivalent(a: str, b: str, answer_type: str = "") -> bool:
    a_norm = _d_lexical_norm(str(a or ""))
    b_norm = _d_lexical_norm(str(b or ""))
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True

    if answer_type in {"country", "nationality"}:
        if _d_country_alias_key(a) == _d_country_alias_key(b):
            return True

    a_key = _d_answer_alias_key(a)
    b_key = _d_answer_alias_key(b)
    if a_key and b_key and a_key == b_key:
        return True

    # Conservative same-name guard for obvious expanded/shortened person names.
    if answer_type == "person":
        a_tokens = a_key.split()
        b_tokens = b_key.split()
        if len(a_tokens) >= 2 and len(b_tokens) >= 2:
            a_set = set(a_tokens)
            b_set = set(b_tokens)
            if a_set.issubset(b_set) or b_set.issubset(a_set):
                a_numbers = {
                    token for token in a_set if any(ch.isdigit() for ch in token)
                }
                b_numbers = {
                    token for token in b_set if any(ch.isdigit() for ch in token)
                }
                if a_numbers == b_numbers:
                    return True
    return False


def _d_fake_answer_type_ok(value: str, answer_type: str) -> bool:
    value = str(value or "").strip()
    if not value or "\n" in value or len(value) > 120:
        return False
    norm = _normalize_surface_text(value)
    if not norm:
        return False

    if answer_type == "yes_no":
        return norm in {"yes", "no"}
    if answer_type == "year":
        return bool(re.fullmatch(r"\d{3,4}", value.strip()))
    if answer_type == "date":
        return bool(re.search(r"\d", value)) and len(value) <= 50
    if answer_type == "number":
        return bool(re.fullmatch(r"[+-]?[\d,.]+(?:\s*%|\s*[A-Za-z]+)?", value))
    if answer_type in {
        "person",
        "country",
        "nationality",
        "location",
        "work",
        "organization",
        "other",
    }:
        if norm in {"yes", "no", "true", "false"}:
            return False
        return any(ch.isalpha() for ch in value)
    return any(ch.isalnum() for ch in value)


def _d_pair_fake_answer(
    plan: Dict[str, Any],
    canonical_answer: str,
    aliases: Iterable[str],
) -> Tuple[str, str]:
    pair = [str(item) for item in plan.get("pair", [])]
    if len(pair) != 2:
        return "", ""
    answer_type = str(plan.get("answer_type", ""))
    gold_forms = [canonical_answer, *[str(alias) for alias in aliases or []]]

    first_is_gold = any(
        _d_answers_equivalent(pair[0], gold, answer_type) for gold in gold_forms
    )
    second_is_gold = any(
        _d_answers_equivalent(pair[1], gold, answer_type) for gold in gold_forms
    )
    if first_is_gold and not second_is_gold:
        return pair[1], pair[0]
    if second_is_gold and not first_is_gold:
        return pair[0], pair[1]
    return "", ""


def _d_date_phrase(value: str) -> str:
    value = str(value or "").strip()
    if re.fullmatch(r"\d{3,4}", value):
        return f"in {value}"
    return f"on {value}"


def _d_render_direct_sentence(
    plan: Dict[str, Any],
    fake_answer: str,
    counterpart: str = "",
) -> str:
    relation = str(plan.get("relation_key", ""))
    fake = str(fake_answer).strip()
    other = str(counterpart).strip()

    pair_templates = {
        "born_earlier": "{fake} was born earlier than {other}.",
        "born_later": "{fake} was born later than {other}.",
        "died_earlier": "{fake} died earlier than {other}.",
        "died_later": "{fake} died later than {other}.",
        "younger": "{fake} is younger than {other}.",
        "older": "{fake} is older than {other}.",
        "released_earlier": "{fake} was released earlier than {other}.",
        "released_later": "{fake} was released later than {other}.",
        "released_more_recently": "{fake} was released more recently than {other}.",
        "director_born_earlier": (
            "The director of {fake} was born earlier than the director of {other}."
        ),
        "director_born_later": (
            "The director of {fake} was born later than the director of {other}."
        ),
    }
    if relation in pair_templates:
        if not fake or not other:
            return ""
        return pair_templates[relation].format(fake=fake, other=other)

    if relation == "same_country":
        pair = plan.get("pair", [])
        if len(pair) != 2:
            return ""
        if _normalize_surface_text(fake) == "yes":
            return f"{pair[0]} and {pair[1]} are from the same country."
        if _normalize_surface_text(fake) == "no":
            return f"{pair[0]} and {pair[1]} are from different countries."
        return ""

    if relation == "same_nationality":
        pair = plan.get("pair", [])
        if len(pair) != 2:
            return ""
        if _normalize_surface_text(fake) == "yes":
            return f"{pair[0]} and {pair[1]} have the same nationality."
        if _normalize_surface_text(fake) == "no":
            return f"{pair[0]} and {pair[1]} have different nationalities."
        return ""

    subject = str(plan.get("subject", "")).strip()
    work = str(plan.get("work", "")).strip()

    if relation == "relative_death_date":
        relative = str(plan.get("relative", "")).strip()
        if not subject or not relative:
            return ""
        return f"{subject}'s {relative} died {_d_date_phrase(fake)}."
    if relation == "death_date":
        return f"{subject} died {_d_date_phrase(fake)}." if subject else ""
    if relation == "birth_date":
        return f"{subject} was born {_d_date_phrase(fake)}." if subject else ""
    if relation == "birth_place":
        return f"{subject} was born in {fake}." if subject else ""
    if relation == "spouse_of_director":
        return f"The spouse of the director of {work} is {fake}." if work else ""
    if relation == "spouse":
        return f"{subject}'s spouse is {fake}." if subject else ""
    if relation in {
        "father",
        "mother",
        "paternal_grandfather",
        "maternal_grandfather",
        "paternal_grandmother",
        "maternal_grandmother",
    }:
        readable = relation.replace("_", " ")
        return f"{subject}'s {readable} is {fake}." if subject else ""
    if relation == "performer_country":
        return f"The performer of the song {work} is from {fake}." if work else ""
    if relation == "country":
        return f"{subject} is from {fake}." if subject else ""
    if relation == "nationality":
        return f"{subject}'s nationality is {fake}." if subject else ""
    if relation == "director":
        return f"{fake} directed {work}." if work else ""
    if relation == "author":
        return f"{fake} wrote {work}." if work else ""
    if relation == "release_year":
        return f"{work} was released in {fake}." if work else ""
    if relation == "release_date":
        return f"{work} was released {_d_date_phrase(fake)}." if work else ""
    return ""


def _d_non_answer_role_values(
    plan: Dict[str, Any],
    sources: Iterable[Dict[str, Any]],
    answer_type: str,
) -> List[str]:
    """
    Collect entities that already occupy a non-answer role in the gold reasoning chain.

    This is deliberately conservative and is currently applied only to person-valued
    open-relation questions. In 2Wiki-style multi-hop questions, gold support titles
    commonly name the subject/work and the bridge entity (for example, a film and its
    director). Reusing one of those entities as the fake answer can create impossible
    self-relations such as "the director's spouse is the director".

    The values are used only as a rejection blacklist for fake-answer candidates; no
    gold evidence is modified.
    """
    if str(answer_type or "") != "person":
        return []

    values: List[str] = []
    for key in ("subject", "work"):
        value = str(plan.get(key, "") or "").strip()
        if value:
            values.append(value)

    # For open person-valued multi-hop questions, supporting titles typically encode
    # the subject/work and bridge-role entities. A fake answer matching any of them is
    # therefore a role collision rather than a clean alternative answer. Pairwise
    # comparison questions never use this candidate-generation path.
    for source in sources:
        title = str(source.get("title", "") or "").strip()
        if title:
            values.append(title)

    unique: List[str] = []
    seen: Set[str] = set()
    for value in values:
        key = _d_answer_alias_key(value) or _d_lexical_norm(value)
        if key and key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _d_fake_answer_has_role_collision(
    fake_answer: str,
    forbidden_role_values: Iterable[str],
    answer_type: str,
) -> bool:
    """Return True when the fake answer reuses a known non-answer role entity."""
    return any(
        _d_answers_equivalent(fake_answer, role_value, answer_type)
        for role_value in forbidden_role_values
        if str(role_value or "").strip()
    )


def _d_choose_anchor_source(
    sources: List[Dict[str, Any]],
    preferred_text: str = "",
) -> Optional[Dict[str, Any]]:
    if not sources:
        return None
    preferred = _d_lexical_norm(preferred_text)
    if preferred:
        for source in reversed(sources):
            title = _d_lexical_norm(str(source.get("title", "")))
            if title and (
                title == preferred or title in preferred or preferred in title
            ):
                return source
    # Preserve the previous D preference for the last eligible gold support, which is
    # usually compatible with M because M preferentially removes the first support.
    return sources[-1]


def inject_distractor(
    trace: Dict[str, Any],
    excluded_support_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Direct Relation-Matched Distracting Answer Sentence (D v7.2).

    The LLM is no longer allowed to decide the logical relation expressed by the
    distractor and is no longer used as a self-verifier. Python first parses the exact
    question into a controlled relation schema. For comparison and yes/no questions,
    the wrong answer is derived deterministically from the question entities and gold
    answer. For open-valued relations, the LLM proposes only short fake answer values;
    Python renders the final distractor with a fixed relation-matched template.

    This guarantees that the generated sentence literally states the requested relation
    toward the wrong answer (e.g. earlier/later, younger/older, release order, spouse,
    parent/grandparent, death date, country/nationality). For person-valued open relations,
    v7.2 keeps the subject/work/bridge-role collision guard and adds a controlled
    LLM fallback that may only map paraphrases to existing deterministic relation schemas. Unknown relation forms still abstain; no free-form distractor sentence is allowed.

    All original gold evidence remains untouched. One non-supporting passage is replaced
    by the synthetic distractor at rank 2, so D remains introduced conflicting evidence
    rather than corruption of gold evidence.
    """
    result = deepcopy(trace)
    excluded = {str(item) for item in (excluded_support_ids or [])}
    supporting, non_supporting, _ = _split_documents(result)
    supporting.sort(key=lambda item: int(item.get("rank", 999999)))
    non_supporting.sort(key=lambda item: int(item.get("rank", 999999)))

    source_candidates = [
        item for item in supporting if str(item.get("passage_id", "")) not in excluded
    ]
    if not source_candidates:
        raise ValueError("No supporting passage is available for D injection.")
    if not non_supporting:
        raise ValueError("No non-supporting passage is available for D injection.")

    question_text = str(result.get("question", ""))
    plan = _d_relation_plan(question_text)
    relation_plan_source = "deterministic_regex"

    if not plan:
        plan = _d_relation_plan_fallback(question_text)
        relation_plan_source = (
            "controlled_llm_existing_schema_fallback" if plan else "none"
        )

    if not plan:
        raise ValueError("D_no_supported_direct_relation_template")

    canonical_answer = str(result.get("canonical_answer", "")).strip()
    aliases = [str(alias) for alias in (result.get("answer_aliases", []) or [])]
    answer_type = str(plan.get("answer_type", "other"))
    relation_key = str(plan.get("relation_key", ""))

    fake_answer = ""
    counterpart = ""
    candidate_count = 0
    selected_candidate_rank = 0

    pair_relations = {
        "born_earlier",
        "born_later",
        "died_earlier",
        "died_later",
        "younger",
        "older",
        "released_earlier",
        "released_later",
        "released_more_recently",
        "director_born_earlier",
        "director_born_later",
    }

    if relation_key in pair_relations:
        fake_answer, counterpart = _d_pair_fake_answer(plan, canonical_answer, aliases)
        if not fake_answer:
            raise ValueError("D_gold_answer_does_not_match_pair_entities")
        candidate_count = 1
        selected_candidate_rank = 1

    elif answer_type == "yes_no":
        canonical_norm = _normalize_surface_text(canonical_answer)
        if canonical_norm in {"yes", "true"}:
            fake_answer = "no"
        elif canonical_norm in {"no", "false"}:
            fake_answer = "yes"
        else:
            raise ValueError("D_binary_question_without_binary_gold_answer")
        candidate_count = 1
        selected_candidate_rank = 1

    else:
        formatted_sources = "\n\n".join(
            f"P{index}\nTitle: {source.get('title', '')}\nText: {source.get('text', '')}"
            for index, source in enumerate(reversed(source_candidates), start=1)
        )
        proposer = llm.with_structured_output(
            DistractorFakeAnswerProposal,
            method="json_schema",
        )
        proposal_prompt = f"""
Propose up to FIVE plausible WRONG ANSWER VALUES only. Do NOT write a distractor sentence.

Question:
{result.get('question', '')}

Gold answer:
{canonical_answer}

Required relation schema:
{relation_key}

Required answer type:
{answer_type}

Gold supporting passages:
{formatted_sources}

Rules:
- Each fake_answer must be a short direct answer to the exact Question.
- It must be clearly different in meaning from the Gold answer and its obvious aliases.
- It must have the required answer type: {answer_type}.
- It must be plausible enough to appear in a retrieved evidence sentence.
- Do not return an explanation, sentence, relation, or multiple facts as fake_answer.
- For dates/years/numbers, return only the value.
- For people/countries/locations/works/organizations, return only the name/value.
- Do not use a longer/shorter spelling of the same person or place as the Gold answer.
- For person-valued multi-hop relations, do NOT return the question subject, work,
  director/performer, parent, or another bridge-role entity already named by the gold
  supporting passages. The fake answer must be a genuinely different person.

Return can_inject=true whenever at least one clean wrong value exists. Order candidates
strongest-first. Return at most 5 candidates.
"""
        proposal = invoke_structured(proposer, proposal_prompt)
        candidates = proposal.candidates[:5] if proposal.can_inject else []
        candidate_count = len(candidates)
        if not candidates:
            raise ValueError("D_no_fake_answer_candidates_proposed")

        gold_forms = [canonical_answer, *aliases]
        forbidden_role_values = _d_non_answer_role_values(
            plan, source_candidates, answer_type
        )
        for rank, candidate in enumerate(candidates, start=1):
            value = str(candidate.fake_answer or "").strip()
            if not _d_fake_answer_type_ok(value, answer_type):
                continue
            if any(
                _d_answers_equivalent(value, gold, answer_type) for gold in gold_forms
            ):
                continue
            # v7.1: do not reuse the subject/work/bridge entity as the fake answer.
            # This blocks impossible self-relations such as a director being selected
            # as the spouse of that same director, while preserving v7's templates.
            if _d_fake_answer_has_role_collision(
                value, forbidden_role_values, answer_type
            ):
                continue
            fake_answer = value
            selected_candidate_rank = rank
            break

        if not fake_answer:
            raise ValueError("D_no_valid_fake_answer_value_found")

    if not _d_fake_answer_type_ok(fake_answer, answer_type):
        raise ValueError("D_fake_answer_type_guard_failed")
    if any(
        _d_answers_equivalent(fake_answer, gold, answer_type)
        for gold in [canonical_answer, *aliases]
    ):
        raise ValueError("D_fake_answer_equivalent_to_gold")

    distractor_sentence = _d_render_direct_sentence(plan, fake_answer, counterpart)
    if not distractor_sentence:
        raise ValueError("D_direct_relation_render_failed")
    if "\n" in distractor_sentence or len(distractor_sentence) > 420:
        raise ValueError("D_direct_relation_sentence_guard_failed")

    fake_norm = _normalize_surface_text(fake_answer)
    sentence_norm = _normalize_surface_text(distractor_sentence)
    if fake_norm not in sentence_norm:
        raise ValueError("D_fake_answer_not_explicit_in_sentence")

    preferred_anchor = (
        fake_answer
        if relation_key in pair_relations
        else str(plan.get("anchor_hint", plan.get("subject", plan.get("work", ""))))
    )
    chosen_source = _d_choose_anchor_source(source_candidates, preferred_anchor)
    if chosen_source is None:
        raise ValueError("D_no_anchor_source_available")

    replacement = non_supporting[-1]
    source_id = str(chosen_source["passage_id"])
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
            "title": chosen_source.get("title", ""),
            "passage_id": distractor_id,
            "rank": DISTRACTOR_RANK,
            "text": distractor_sentence,
            "is_supporting": False,
            "document_role": "distractor",
            "source_passage_id": source_id,
            "anchor_passage_id": source_id,
            "replaced_passage_id": replacement_id,
        },
    )
    _rerank(documents)
    result["retrieval_events"] = documents
    result.setdefault("failure_history", []).append(
        {
            "failure": "D",
            "intervention_version": D_INTERVENTION_VERSION,
            "failure_type": "direct_relation_matched_distracting_answer_sentence",
            "source_passage_id": source_id,
            "anchor_passage_id": source_id,
            "replaced_passage_id": replacement_id,
            "distractor_passage_id": distractor_id,
            "relation_key": relation_key,
            "relation_plan_source": relation_plan_source,
            "validation_mode": "deterministic_relation_template",
            "fake_answer": fake_answer,
            "proposed_wrong_answer": fake_answer,
            "fact_type": answer_type,
            "original_claim": f"Gold answer: {canonical_answer}",
            "modified_claim": distractor_sentence,
            "original_passage_text": str(chosen_source.get("text", "")),
            "distractor_sentence": distractor_sentence,
            "distractor_text": distractor_sentence,
            "proposal_candidate_count": candidate_count,
            "selected_candidate_rank": selected_candidate_rank,
            "distractor_rank": DISTRACTOR_RANK,
        }
    )
    return result


def _format_gold_passages_for_l(
    documents: Iterable[Dict[str, Any]],
) -> str:
    return "\n\n".join(
        f"Gold passage ID: {document.get('passage_id', '')}\n"
        f"Gold passage title: {document.get('title', '')}\n"
        f"Gold passage text: {document.get('text', '')}"
        for document in documents
    )


def _normalize_surface_text(text: str) -> str:
    """
    Normalize harmless formatting differences so formatting-only LLM2 output
    is treated as unchanged.

    This intentionally ignores:
    - whitespace changes
    - quote style changes
    - dash style changes
    - capitalization-only changes

    It does NOT remove normal letters, digits, or punctuation content.
    """
    value = str(text or "")
    value = (
        value.replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
        .replace("‑", "-")
        .replace("−", "-")
    )
    value = re.sub(r"\s+", "", value)
    return value.lower()


def _is_generic_bridge_phrase(phrase: str) -> bool:
    """
    Detect generic role/type phrases that are risky for semantic matching.

    These are removed only when LLM1 also supplied at least one more
    identity-bearing phrase.
    """
    raw = str(phrase or "").strip()
    if not raw:
        return True

    lowered = re.sub(r"\s+", " ", raw.lower()).strip()

    generic_exact = {
        "grand duke",
        "the grand duke",
        "the director",
        "the singer",
        "the film",
        "the movie",
        "the settlement",
        "the organization",
        "the actor",
        "the actress",
        "the person",
        "the individual",
        "the duke",
        "the player",
        "the athlete",
        "the politician",
        "the peer",
        "the village",
        "the place",
        "the work",
        "the song",
        "the book",
        "the author",
        "the producer",
        "the musician",
    }
    if lowered in generic_exact:
        return True

    # Also catch descriptions such as "the Chinese film",
    # "the British peer", "a Brazilian player", etc.
    tokens = lowered.split()
    if len(tokens) >= 2 and tokens[0] in {"the", "a", "an"}:
        generic_heads = {
            "director",
            "singer",
            "film",
            "movie",
            "settlement",
            "organization",
            "actor",
            "actress",
            "person",
            "individual",
            "duke",
            "player",
            "athlete",
            "politician",
            "peer",
            "village",
            "place",
            "work",
            "song",
            "book",
            "author",
            "producer",
            "musician",
        }
        if tokens[-1] in generic_heads:
            return True

    return False


def _prefer_identity_bearing_phrases(phrases: List[str]) -> List[str]:
    """
    If LLM1 supplied at least one non-generic identity phrase, drop generic
    role/type phrases before sending the list to LLM2.

    Example:
        ["Zhang Yang", "the director"] -> ["Zhang Yang"]

    If every phrase is generic, preserve the original list rather than
    creating a hard skip gate.
    """
    cleaned = [str(item or "").strip() for item in phrases if str(item or "").strip()]
    identity_bearing = [item for item in cleaned if not _is_generic_bridge_phrase(item)]

    if identity_bearing:
        return identity_bearing
    return cleaned


def _l_normalize_identity(text: str) -> str:
    """Normalize identity text only for safe equality checks."""
    value = str(text or "").strip().lower()
    value = value.replace("’", "'").replace("‘", "'")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s'-]", "", value)
    return value.strip()


def _l_title_variants(title: str) -> List[str]:
    """
    Return only conservative exact textual variants for a gold passage title.

    v17 intentionally does NOT invent aliases, surnames, role words, related
    entity names, or longer/shorter semantic paraphrases.  The only automatic
    variant is removal of a trailing Wikipedia-style disambiguation
    parenthesis, e.g. "John Smith (director)" -> "John Smith".
    """
    raw = str(title or "").strip()
    if not raw:
        return []

    variants = [raw]
    stripped = re.sub(r"\s*\([^()]+\)\s*$", "", raw).strip()
    if stripped and stripped != raw:
        variants.append(stripped)

    seen: Set[str] = set()
    result: List[str] = []
    for item in variants:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _l_exact_pattern(phrase: str) -> re.Pattern:
    """Case-insensitive exact phrase pattern with word-safe outer boundaries."""
    escaped = re.escape(str(phrase or ""))
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


def _l_present_variants(text: str, variants: Iterable[str]) -> List[str]:
    """Return conservative title variants that occur literally in text."""
    found: List[str] = []
    for variant in variants:
        if variant and _l_exact_pattern(variant).search(str(text or "")):
            found.append(variant)
    return found


def _l_identity_is_answer(trace: Dict[str, Any], identity: str) -> bool:
    """Never anonymize the canonical answer itself or an exact answer alias."""
    identity_key = _l_normalize_identity(identity)
    if not identity_key:
        return True

    answers = [trace.get("canonical_answer", ""), *trace.get("answer_aliases", [])]
    return any(
        identity_key == _l_normalize_identity(answer)
        for answer in answers
        if str(answer or "").strip()
    )


def _l_answer_surface_signature(
    trace: Dict[str, Any],
    text: str,
) -> Dict[str, int]:
    """
    Count literal canonical-answer/alias surfaces in a passage.

    L is allowed to obscure only the bridge identity.  If an answer surface
    disappears as a side effect, the candidate is rejected.
    """
    signature: Dict[str, int] = {}
    forms = [trace.get("canonical_answer", ""), *trace.get("answer_aliases", [])]
    for form in forms:
        value = str(form or "").strip()
        if not value or len(value) > 180:
            continue
        key = _l_normalize_identity(value)
        if not key or key in signature:
            continue
        signature[key] = len(list(_l_exact_pattern(value).finditer(str(text or ""))))
    return signature


def _l_has_named_suffix(text: str, match_end: int) -> bool:
    """
    Reject an occurrence when the matched bridge is visibly the prefix of a
    longer named expression.

    Examples rejected:
      Yakuza -> "Yakuza Kiwami", "Yakuza 0"
      Enron  -> "Enron Energy Services", "Enron Xcelerator"
      Universal Soldier -> "Universal Soldier: Regeneration"

    Lower-case role words are not treated as named extensions, so phrases such
    as "Enron executive" or "Aqua song" are not rejected by this guard.
    """
    tail = str(text or "")[match_end:]

    next_token = re.match(
        r"\s+([^\W\d_][\w'’.-]*|\d+)",
        tail,
        flags=re.UNICODE,
    )
    if next_token:
        token = next_token.group(1)
        if token.isdigit() or (token and token[0].isupper()):
            return True

    subtitle_token = re.match(
        r"\s*[:\-–—]\s*([^\W\d_][\w'’.-]*|\d+)",
        tail,
        flags=re.UNICODE,
    )
    if subtitle_token:
        token = subtitle_token.group(1)
        if token.isdigit() or (token and token[0].isupper()):
            return True

    return False


def _l_unique_exact_mention(
    text: str,
    bridge_identity: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Locate exactly ONE unambiguous bridge mention in a target passage.

    This is the central v17 precision restriction:
    - only the exact passage title or its parenthetical-stripped form;
    - no surname expansion;
    - no related-entity expansion;
    - no generic role expansion;
    - no multiple same-surface occurrences;
    - no occurrence embedded in a longer named expression.

    Requiring one unique mention prevents collateral masking when the same
    surface string names a different work, organization, relative, album,
    location, etc. elsewhere in the passage.
    """
    value = str(text or "")
    variants = sorted(_l_title_variants(bridge_identity), key=len, reverse=True)
    if not value or not variants:
        return None, "L_empty_identity_or_target"

    mentions: List[Dict[str, Any]] = []

    for variant in variants:
        for match in _l_exact_pattern(variant).finditer(value):
            start, end = match.span()

            # A shorter variant may overlap the same occurrence already found
            # through a longer title variant. Count that as one mention.
            if any(
                not (end <= item["start"] or start >= item["end"])
                for item in mentions
            ):
                continue

            if _l_has_named_suffix(value, end):
                return None, "L_identity_surface_embedded_in_longer_name"

            mentions.append(
                {
                    "phrase": match.group(0),
                    "variant": variant,
                    "start": start,
                    "end": end,
                }
            )

    if not mentions:
        return None, "L_no_exact_identity_mention"
    if len(mentions) != 1:
        return None, "L_identity_surface_not_unique"

    return mentions[0], ""


def _l_replace_one_exact_mention(
    text: str,
    mention: Dict[str, Any],
    replacement: str,
) -> str:
    """Replace exactly the one pre-validated bridge span."""
    value = str(text or "")
    start = int(mention["start"])
    end = int(mention["end"])
    return value[:start] + replacement + value[end:]


def _l_select_gold_link(
    trace: Dict[str, Any],
    supporting: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Select one high-precision gold linkage.

    Route A — hidden passage -> passage bridge (preferred)
        A supporting passage title/entity is NOT named in the question, occurs
        in its own source passage, and occurs exactly once in a different gold
        passage.  The unique target occurrence is anonymized.

    Route B — question -> passage identity link (strict fallback)
        A gold passage title/entity is named in the question and occurs exactly
        once in that passage text.  To prevent title metadata from leaking the
        identity, v17 anonymizes BOTH the target passage title and the one exact
        text occurrence.

    No semantic alias generation, surname expansion, role-word masking, or
    free-form rewriting is permitted.
    """
    question = str(trace.get("question", ""))

    # ------------------------------------------------------------------
    # A. Hidden passage -> passage bridge.
    # ------------------------------------------------------------------
    p2p_candidates: List[Tuple[int, int, Dict[str, Any]]] = []

    for source_index, source in enumerate(supporting):
        bridge_identity = str(source.get("title", "")).strip()
        if not bridge_identity or _l_identity_is_answer(trace, bridge_identity):
            continue
        if re.search(r"\d", bridge_identity):
            continue
        if _is_generic_bridge_phrase(bridge_identity):
            continue

        variants = _l_title_variants(bridge_identity)
        if not variants:
            continue

        # Strict hidden-bridge requirement: if the identity is already explicit
        # in the question, it is not a passage-to-passage hidden bridge.
        if _l_present_variants(question, variants):
            continue

        # The source title must ground to its own passage text.
        source_text = str(source.get("text", ""))
        if not _l_present_variants(source_text, variants):
            continue

        for target_index, target in enumerate(supporting):
            if source_index == target_index:
                continue

            # Do not construct a pseudo-link between duplicate-title supports.
            if _l_normalize_identity(source.get("title", "")) == _l_normalize_identity(
                target.get("title", "")
            ):
                continue

            mention, reason = _l_unique_exact_mention(
                str(target.get("text", "")),
                bridge_identity,
            )
            if mention is None:
                continue

            candidate = {
                "link_type": "passage_to_passage",
                "source": source,
                "target": target,
                "bridge_identity": bridge_identity,
                "matched_phrase": str(mention["phrase"]),
                "mention": mention,
                "modify_target_title": False,
                "broken_link_description": (
                    f"Anonymizes the unique explicit gold bridge identity "
                    f"'{bridge_identity}' in one supporting passage, breaking "
                    "its hidden passage-to-passage connection while leaving "
                    "all other evidence unchanged."
                ),
            }

            p2p_candidates.append(
                (
                    int(target.get("rank", 999999)),
                    int(source.get("rank", 999999)),
                    candidate,
                )
            )

    if p2p_candidates:
        p2p_candidates.sort(key=lambda item: item[:2])
        return p2p_candidates[0][2], ""

    # ------------------------------------------------------------------
    # B. Strict question -> passage identity link.
    #
    # Unlike v16, masking text alone is not considered sufficient because the
    # retrieval title can still reveal the question identity.  Therefore this
    # route is allowed only when the title can also be anonymized safely.
    # ------------------------------------------------------------------
    q2p_candidates: List[Tuple[int, Dict[str, Any]]] = []

    for target in supporting:
        bridge_identity = str(target.get("title", "")).strip()
        if not bridge_identity or _l_identity_is_answer(trace, bridge_identity):
            continue
        if re.search(r"\d", bridge_identity):
            continue
        if _is_generic_bridge_phrase(bridge_identity):
            continue

        variants = _l_title_variants(bridge_identity)
        if not variants or not _l_present_variants(question, variants):
            continue

        # If an answer surface is literally present in the title, replacing
        # that title could delete answer evidence rather than only linkage.
        title_answer_signature = _l_answer_surface_signature(trace, bridge_identity)
        if any(count > 0 for count in title_answer_signature.values()):
            continue

        mention, reason = _l_unique_exact_mention(
            str(target.get("text", "")),
            bridge_identity,
        )
        if mention is None:
            continue

        candidate = {
            "link_type": "question_to_passage",
            "source": None,
            "target": target,
            "bridge_identity": bridge_identity,
            "matched_phrase": str(mention["phrase"]),
            "mention": mention,
            "modify_target_title": True,
            "broken_link_description": (
                f"Anonymizes the unique gold identity '{bridge_identity}' in "
                "both the supporting passage title and its one exact text "
                "mention, breaking the explicit question-to-evidence identity "
                "link while preserving all non-identity factual content."
            ),
        }
        q2p_candidates.append((int(target.get("rank", 999999)), candidate))

    if q2p_candidates:
        q2p_candidates.sort(key=lambda item: item[0])
        return q2p_candidates[0][1], ""

    return None, "L_no_strict_unique_gold_link_found"


def inject_link_failure(
    trace: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Gold-Link Entity Anonymization (v17 strict exact-link).

    Exactly one answer-relevant identity surface is obscured with ``ENTITY_A``.
    No LLM is used and no sentence is rewritten.

    Precision restrictions:
    - exact title identity only (plus parenthetical-stripped title);
    - exactly one target-text occurrence;
    - no surname/alias/role expansion;
    - no longer-name-prefix occurrence;
    - canonical answer surfaces must be unchanged;
    - all numeric content must be unchanged;
    - passage->passage links must be hidden from the question;
    - question->passage links also anonymize the target title to prevent title
      metadata from preserving the supposedly broken link.

    The injector abstains whenever these conditions are not satisfied.
    """
    result = deepcopy(trace)
    supporting, _, _ = _split_documents(result)
    supporting.sort(key=lambda item: int(item.get("rank", 999999)))

    if not (2 <= len(supporting) <= 4):
        return None, "L_requires_two_to_four_retrieved_supporting_passages"

    plan, reason = _l_select_gold_link(result, supporting)
    if plan is None:
        return None, reason

    target = plan["target"]
    source = plan.get("source")
    bridge_identity = str(plan["bridge_identity"])
    matched_phrase = str(plan["matched_phrase"])
    mention = dict(plan["mention"])
    link_type = str(plan["link_type"])
    broken_link_description = str(plan["broken_link_description"])
    modify_target_title = bool(plan.get("modify_target_title", False))

    target_id = str(target.get("passage_id", ""))
    target_label = f"P{supporting.index(target) + 1}"
    source_id = str(source.get("passage_id", "")) if source is not None else ""
    source_label = f"P{supporting.index(source) + 1}" if source is not None else ""

    original_text = str(target.get("text", ""))
    original_title = str(target.get("title", ""))
    replacement = "ENTITY_A"

    if not original_text.strip():
        return None, "L_empty_target_passage"
    if replacement.casefold() in original_text.casefold():
        return None, "L_replacement_token_already_present"
    if modify_target_title and replacement.casefold() in original_title.casefold():
        return None, "L_replacement_token_already_present_in_title"

    answer_signature_before = _l_answer_surface_signature(result, original_text)

    corrupted_text = _l_replace_one_exact_mention(
        original_text,
        mention,
        replacement,
    )
    corrupted_title = replacement if modify_target_title else original_title

    if corrupted_text == original_text:
        return None, "L_no_verified_identity_span_replaced"

    # No other exact title form may remain in the target text.  Because the
    # selector required a unique occurrence, any residual variant means the
    # replacement did not behave as expected.
    if any(
        _l_exact_pattern(variant).search(corrupted_text)
        for variant in _l_title_variants(bridge_identity)
    ):
        return None, "L_identity_surface_remained_after_replacement"

    # Protect answer evidence.
    answer_signature_after = _l_answer_surface_signature(result, corrupted_text)
    if answer_signature_before != answer_signature_after:
        return None, "L_answer_surface_changed"

    # Preserve every Arabic-number string exactly.
    if re.findall(r"\d+(?:[.,:/-]\d+)*", original_text) != re.findall(
        r"\d+(?:[.,:/-]\d+)*", corrupted_text
    ):
        return None, "L_numeric_content_changed"

    # Exactly one text identity span is allowed to change.
    if corrupted_text.count(replacement) != 1:
        return None, "L_replacement_count_not_exactly_one"

    # Apply the already-validated changes.
    target["text"] = corrupted_text
    target["title"] = corrupted_title

    source_ids = [source_id] if source_id else []
    replaced_phrases = [matched_phrase]
    proposed_link_phrase = matched_phrase

    result["l_target_passage_id"] = target_id
    result["l_source_passage_ids"] = source_ids
    result["l_source_passage_label"] = source_label
    result["l_target_passage_label"] = target_label
    result["l_bridge_identity"] = bridge_identity
    result["l_proposed_bridge_phrases"] = replaced_phrases
    result["l_broken_link_description"] = broken_link_description
    result["l_link_type"] = link_type
    result["l_original_link_phrase"] = proposed_link_phrase
    result["l_original_link_phrases"] = replaced_phrases
    result["l_replacement_phrase"] = replacement
    result["l_original_passage_text"] = original_text
    result["l_corrupted_passage_text"] = corrupted_text
    result["l_original_target_title"] = original_title
    result["l_corrupted_target_title"] = corrupted_title
    result["l_title_modified"] = modify_target_title
    result["l_validation_mode"] = "strict_unique_exact_identity_v17"
    result["l_supporting_count_at_injection"] = len(supporting)

    result.setdefault("failure_history", []).append(
        {
            "failure": "L",
            "intervention_version": L_INTERVENTION_VERSION,
            "failure_type": "gold_link_entity_anonymized",
            "link_type": link_type,
            "validation_mode": "strict_unique_exact_identity_v17",
            "supporting_count_at_injection": len(supporting),
            "source_passage_ids": source_ids,
            "target_passage_id": target_id,
            "source_passage_label": source_label,
            "target_passage_label": target_label,
            "bridge_identity": bridge_identity,
            "proposed_bridge_phrases": replaced_phrases,
            "broken_link_description": broken_link_description,
            "original_link_phrase": proposed_link_phrase,
            "original_link_phrases": replaced_phrases,
            "replacement_phrase": replacement,
            "original_passage_text": original_text,
            "corrupted_passage_text": corrupted_text,
            "original_target_title": original_title,
            "corrupted_target_title": corrupted_title,
            "title_modified": modify_target_title,
            "replacement_count": 1,
            "evidence_modified": True,
            "strict_identity_guard": "passed",
        }
    )

    return result, ""

def inject_single(
    trace: Dict[str, Any], failure: str
) -> Tuple[Optional[Dict[str, Any]], str]:
    eligible, reason = failure_eligibility(trace, failure)
    if not eligible:
        return None, reason

    if failure == "M":
        return _finalize(inject_missing_evidence(trace), ["M"]), ""
    if failure == "D":
        try:
            return _finalize(inject_distractor(trace), ["D"]), ""
        except ValueError as error:
            reason = str(error)
            if reason.startswith("D_"):
                return None, reason
            raise
    if failure == "L":
        result, reason = inject_link_failure(trace)
        if result is None:
            return None, reason
        return _finalize(result, ["L"]), ""
    raise ValueError(f"Unsupported failure type: {failure}")


def _add_saved_m(
    base_trace: Dict[str, Any],
    m_trace: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    result = deepcopy(base_trace)
    m_history = _history_entry(m_trace, "M")
    removed_ids = {
        str(item) for item in m_history.get("removed_passage_ids", []) if item
    }
    current_ids = {
        str(item.get("passage_id", "")) for item in result.get("retrieval_events", [])
    }
    if not removed_ids or not removed_ids.issubset(current_ids):
        return None
    result["retrieval_events"] = [
        item
        for item in result.get("retrieval_events", [])
        if str(item.get("passage_id", "")) not in removed_ids
    ]
    _rerank(result["retrieval_events"])
    result.setdefault("failure_history", []).append(deepcopy(m_history))
    return result


def _add_saved_d(
    base_trace: Dict[str, Any],
    d_trace: Dict[str, Any],
    allow_missing_source: bool = False,
) -> Optional[Dict[str, Any]]:
    result = deepcopy(base_trace)
    d_history = _history_entry(d_trace, "D")
    source_id = str(d_history.get("source_passage_id", ""))
    replaced_id = str(d_history.get("replaced_passage_id", ""))
    current_ids = {
        str(item.get("passage_id", "")) for item in result.get("retrieval_events", [])
    }

    if replaced_id not in current_ids:
        return None
    if not allow_missing_source and source_id not in current_ids:
        return None

    distractor_id = str(d_history.get("distractor_passage_id", ""))
    distractor = next(
        (
            deepcopy(item)
            for item in d_trace.get("retrieval_events", [])
            if str(item.get("passage_id", "")) == distractor_id
        ),
        None,
    )
    if distractor is None:
        return None

    documents = [
        item
        for item in result.get("retrieval_events", [])
        if str(item.get("passage_id", "")) != replaced_id
    ]
    documents.insert(min(DISTRACTOR_RANK - 1, len(documents)), distractor)
    _rerank(documents)
    result["retrieval_events"] = documents
    result.setdefault("failure_history", []).append(deepcopy(d_history))
    return result


def _add_saved_l(
    base_trace: Dict[str, Any],
    l_trace: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Replay one already-validated L intervention exactly.

    v17 may also anonymize the target passage title for strict
    question-to-passage linkage failures, so both text and title are restored
    from the saved L history when present.
    """
    result = deepcopy(base_trace)
    l_history = _history_entry(l_trace, "L")
    target_id = str(l_history.get("target_passage_id", ""))
    corrupted_text = l_history.get("corrupted_passage_text")
    corrupted_title = l_history.get("corrupted_target_title")

    target = next(
        (
            item
            for item in result.get("retrieval_events", [])
            if str(item.get("passage_id", "")) == target_id
        ),
        None,
    )
    if target is None or corrupted_text is None:
        return None

    target["text"] = str(corrupted_text)
    if corrupted_title is not None:
        target["title"] = str(corrupted_title)

    result["l_target_passage_id"] = target_id
    result["l_source_passage_ids"] = [
        str(item) for item in l_history.get("source_passage_ids", [])
    ]
    result["l_broken_link_description"] = str(
        l_history.get("broken_link_description", "") or ""
    )
    result["l_bridge_identity"] = str(l_history.get("bridge_identity", "") or "")
    result["l_source_passage_label"] = str(
        l_history.get("source_passage_label", "") or ""
    )
    result["l_target_passage_label"] = str(
        l_history.get("target_passage_label", "") or ""
    )
    result["l_proposed_bridge_phrases"] = [
        str(item) for item in l_history.get("proposed_bridge_phrases", []) if item
    ]
    result["l_original_link_phrase"] = str(
        l_history.get("original_link_phrase", "") or ""
    )
    result["l_original_link_phrases"] = [
        str(item) for item in l_history.get("original_link_phrases", []) if item
    ]
    result["l_replacement_phrase"] = str(
        l_history.get("replacement_phrase", "") or ""
    )
    result["l_original_passage_text"] = str(
        l_history.get("original_passage_text", "") or ""
    )
    result["l_corrupted_passage_text"] = str(corrupted_text)
    result["l_original_target_title"] = str(
        l_history.get("original_target_title", target.get("title", "")) or ""
    )
    result["l_corrupted_target_title"] = str(
        corrupted_title if corrupted_title is not None else target.get("title", "")
    )
    result["l_title_modified"] = bool(l_history.get("title_modified", False))
    result["l_validation_mode"] = str(
        l_history.get("validation_mode", "") or ""
    )
    result.setdefault("failure_history", []).append(deepcopy(l_history))
    return result

def _m_removed_id(trace: Dict[str, Any]) -> str:
    entry = _history_entry(trace, "M")
    values = entry.get("removed_passage_ids", [])
    return str(values[0]) if values else ""


def _d_source_id(trace: Dict[str, Any]) -> str:
    return str(_history_entry(trace, "D").get("source_passage_id", ""))


def _l_target_id(trace: Dict[str, Any]) -> str:
    return str(_history_entry(trace, "L").get("target_passage_id", ""))


def _run_with_timeout(ctx, pool, function, args):
    job = pool.apply_async(function, args)
    try:
        return job.get(timeout=FAILURE_INJECTION_TIMEOUT_SECONDS), pool, False
    except mp.TimeoutError:
        pool.terminate()
        pool.join()
        pool = ctx.Pool(processes=1)
        return None, pool, True


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


def _skip(
    skipped: List[Dict[str, Any]],
    trace_id: str,
    failures: List[str],
    reason: str,
) -> None:
    skipped.append({"trace_id": trace_id, "failures": failures, "reason": reason})


def main():
    baseline_path = "data/processed/pilot_baseline_results.json"
    with open(baseline_path, "r", encoding="utf-8") as file:
        baseline = json.load(file)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=1)

    single_paths = {
        "M": "data/failures/single/missing_evidence.json",
        "D": "data/failures/single/distractor.json",
        "L": "data/failures/single/linkage.json",
    }
    singles: Dict[str, Dict[str, Dict[str, Any]]] = {
        "M": {},
        "D": {},
        "L": {},
    }
    skip_reasons: Dict[str, Dict[str, str]] = {"M": {}, "D": {}, "L": {}}

    for failure in ["M", "D", "L"]:
        accepted: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        for trace in baseline:
            trace_id = str(trace.get("trace_id", ""))
            try:
                payload, pool, timed_out = _run_with_timeout(
                    ctx,
                    pool,
                    inject_single,
                    (trace, failure),
                )
                if timed_out:
                    result = None
                    reason = "failure_injection_timeout"
                else:
                    result, reason = payload
            except Exception as error:
                result = None
                reason = f"failure_injection_error:{type(error).__name__}:{error}"
            if result is None:
                skip_reasons[failure][trace_id] = reason
                _skip(skipped, trace_id, [failure], reason)
            else:
                singles[failure][trace_id] = result
                accepted.append(result)
        _save(single_paths[failure], accepted, skipped)

    baseline_map = {str(trace.get("trace_id", "")): trace for trace in baseline}

    # ------------------------------------------------------------------
    # Compound construction policy
    #
    # Compounds are built sequentially FROM already validated single
    # failure artifacts, rather than requiring the trace to independently
    # exist in every single-failure pool.
    #
    # M_D   : start from every valid single D, then add M.
    # M_L   : start from every valid single L, then add M while protecting L.
    # D_L   : start from every valid single D, then add L.
    # M_D_L : start from every valid D_L compound, then add M while protecting L.
    #
    # This preserves the existing validated base failure exactly and adds
    # only the next intervention on top.
    # ------------------------------------------------------------------

    # -------------------------
    # M_D = single D + add M
    # -------------------------
    accepted, skipped = [], []

    for trace_id, d_trace in singles["D"].items():
        eligible, reason = failure_eligibility(d_trace, "M")
        if not eligible:
            _skip(skipped, trace_id, ["M", "D"], f"M:{reason}")
            continue

        try:
            # D has already replaced a non-supporting passage.
            # Add M by removing one gold supporting passage from that D trace.
            result = inject_missing_evidence(d_trace)
        except Exception as error:
            result = None
            reason = f"M_D_injection_error:{type(error).__name__}:{error}"

        if result is None:
            _skip(skipped, trace_id, ["M", "D"], reason)
            continue

        accepted.append(_finalize(result, ["M", "D"]))

    _save("data/failures/compound/M_D.json", accepted, skipped)

    # -------------------------
    # M_L = single L + add M
    # -------------------------
    accepted, skipped = [], []

    for trace_id, l_trace in singles["L"].items():
        eligible, reason = failure_eligibility(l_trace, "M")
        if not eligible:
            _skip(skipped, trace_id, ["M", "L"], f"M:{reason}")
            continue

        l_target = _l_target_id(l_trace)
        if not l_target:
            _skip(skipped, trace_id, ["M", "L"], "L:missing_target_passage_id")
            continue

        try:
            # Preserve the already-corrupted L target exactly.
            # M must remove a different gold passage.
            result = inject_missing_evidence(l_trace, {l_target})
        except Exception as error:
            result = None
            reason = f"M_L_injection_error:{type(error).__name__}:{error}"

        if result is None:
            _skip(skipped, trace_id, ["M", "L"], reason)
            continue

        accepted.append(_finalize(result, ["M", "L"]))

    _save("data/failures/compound/M_L.json", accepted, skipped)

    # -------------------------
    # D_L = single D + add L
    # -------------------------
    accepted, skipped = [], []
    compound_d_l: Dict[str, Dict[str, Any]] = {}

    for trace_id, d_trace in singles["D"].items():
        eligible, reason = failure_eligibility(d_trace, "L")
        if not eligible:
            _skip(skipped, trace_id, ["D", "L"], f"L:{reason}")
            continue

        try:
            # Keep the validated D distractor exactly as it is.
            # Add L by anonymizing one gold-link identity in the remaining
            # supporting evidence.
            result, reason = inject_link_failure(d_trace)
        except Exception as error:
            result = None
            reason = f"D_L_injection_error:{type(error).__name__}:{error}"

        if result is None:
            _skip(skipped, trace_id, ["D", "L"], reason)
            continue

        finalized = _finalize(result, ["D", "L"])
        compound_d_l[trace_id] = finalized
        accepted.append(finalized)

    _save("data/failures/compound/D_L.json", accepted, skipped)

    # --------------------------------
    # M_D_L = generated D_L + add M
    # --------------------------------
    accepted, skipped = [], []

    for trace_id, d_l_trace in compound_d_l.items():
        eligible, reason = failure_eligibility(d_l_trace, "M")
        if not eligible:
            _skip(skipped, trace_id, ["M", "D", "L"], f"M:{reason}")
            continue

        l_target = _l_target_id(d_l_trace)
        if not l_target:
            _skip(
                skipped,
                trace_id,
                ["M", "D", "L"],
                "L:missing_target_passage_id",
            )
            continue

        try:
            # Always protect the L-corrupted target.
            m_excluded = {l_target}

            # Prefer to preserve D's original gold anchor/source too when
            # another removable gold passage exists. If there are only two
            # gold passages total, allow M to remove the D source so the
            # triple remains constructible; the D distractor itself remains
            # present and unchanged.
            d_source = _d_source_id(d_l_trace)
            supporting, _, _ = _split_documents(d_l_trace)
            removable_non_l = [
                str(item.get("passage_id", ""))
                for item in supporting
                if str(item.get("passage_id", "")) != l_target
            ]

            if d_source and d_source in removable_non_l and len(removable_non_l) > 1:
                m_excluded.add(d_source)

            result = inject_missing_evidence(d_l_trace, m_excluded)

        except Exception as error:
            result = None
            reason = f"M_D_L_injection_error:{type(error).__name__}:{error}"

        if result is None:
            _skip(skipped, trace_id, ["M", "D", "L"], reason)
            continue

        accepted.append(_finalize(result, ["M", "D", "L"]))

    _save("data/failures/compound/M_D_L.json", accepted, skipped)

    pool.close()
    pool.join()


if __name__ == "__main__":
    mp.freeze_support()
    main()