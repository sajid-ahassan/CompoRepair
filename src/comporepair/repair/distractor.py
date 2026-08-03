import json
import re
from copy import deepcopy

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ..pipeline.baseline_rag import generation_llm

load_dotenv()

judge_llm = generation_llm



def format_evidence(documents):
    passages = []
    id_map = {}

    for index, document in enumerate(documents, start=1):
        neutral_id = f"P{index}"
        original_id = str(document.get("passage_id", neutral_id))

        id_map[neutral_id] = original_id

        passages.append(f"{neutral_id}:\n{document.get('text', '')}")

    return "\n\n".join(passages), id_map


def parse_response(response_content):
    default = {
        "conflict_detected": False,
        "remove_passage_id": None,
        "confidence": 0.0,
        "reason": "Parsing failed",
    }

    if not isinstance(response_content, str) or not response_content.strip():
        return default

    try:
        cleaned = re.sub(
            r"```(?:json)?|```",
            "",
            response_content,
            flags=re.IGNORECASE,
        ).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1:
            return default

        result = json.loads(cleaned[start : end + 1])

        passage_id = result.get("remove_passage_id")

        if passage_id is not None:
            passage_id = str(passage_id).strip()

        return {
            "conflict_detected": bool(
                result.get("conflict_detected", False)
            ),
            "remove_passage_id": passage_id,
            "confidence": float(result.get("confidence", 0.0)),
            "reason": str(result.get("reason", "")),
        }

    except (ValueError, TypeError, json.JSONDecodeError):
        return default


def detect_distractor(question, documents):
    evidence, id_map = format_evidence(documents)

    prompt = f"""
You are an evidence-conflict detector.

Use only the provided passages.

Question:
{question}

Passages:
{evidence}

Identify at most ONE passage containing a clearly contradictory or
misleading answer-related claim.

Rules:
- Analyze all passages together.
- Never select both sides of a contradiction.
- Preserve bridge and entity-linking evidence.
- Do not remove a passage merely because it is irrelevant.
- If you cannot determine which single passage is misleading,
  return null.
- Use only neutral IDs such as P1, P2, or null.
- Keep the reason to one short sentence.

Return valid JSON only:

{{
  "conflict_detected": true,
  "remove_passage_id": "P1",
  "confidence": 0.0,
  "reason": "short explanation"
}}
"""

    response = judge_llm.invoke(prompt)
    decision = parse_response(response.content)

    neutral_id = decision["remove_passage_id"]

    decision["remove_passage_id"] = (
        id_map.get(neutral_id)
        if neutral_id in id_map
        else None
    )

    return decision

def repair_distractor(trace):
    repaired_trace = deepcopy(trace)

    documents = repaired_trace.get("retrieval_events", [])

    decision = detect_distractor(
        repaired_trace["question"],
        documents,
    )

    valid_ids = {
        str(document.get("passage_id", ""))
        for document in documents
    }

    requested_id = decision.get("remove_passage_id")

    should_remove = (
        decision.get("conflict_detected", False)
        and decision.get("confidence", 0.0) >= 0.70
        and requested_id is not None
        and requested_id in valid_ids
    )

    remove_ids = {requested_id} if should_remove else set()

    filtered_documents = [
        document
        for document in documents
        if str(document.get("passage_id", "")) not in remove_ids
    ]

    # Reset ranking after removal
    for rank, document in enumerate(filtered_documents, start=1):
        document["rank"] = rank

    repaired_trace["retrieval_events"] = filtered_documents

    history = list(repaired_trace.get("repair_history", []))

    history.append(
        {
            "failure": "D",
            "repair": "global_conflict_filtering",
            "decision": decision,
            "removed_passage_ids": list(remove_ids),
            "unresolved_conflict": (
                decision.get("conflict_detected", False)
                and not should_remove
            ),
        }
    )

    repaired_trace["repair_history"] = history

    return repaired_trace