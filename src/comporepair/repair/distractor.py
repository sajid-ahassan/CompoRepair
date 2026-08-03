import json
import re
from copy import deepcopy

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

judge_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


def format_evidence(documents):
    passages = []
    id_map = {}

    for index, document in enumerate(documents, start=1):
        neutral_id = f"P{index}"
        original_id = str(document.get("passage_id", neutral_id))

        id_map[neutral_id] = original_id

        passages.append(f"{neutral_id}:\n{document.get('text', '')}")

    return "\n\n".join(passages), id_map


def parse_response(response):
    try:
        cleaned = re.sub(
            r"```(?:json)?|```",
            "",
            response,
            flags=re.IGNORECASE,
        ).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        result = json.loads(cleaned[start : end + 1])

        return {
            "conflict_detected": result.get(
                "conflict_detected",
                False,
            ),
            "remove_passage_ids": result.get(
                "remove_passage_ids",
                [],
            ),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": result.get("reason", ""),
        }

    except Exception:
        return {
            "conflict_detected": False,
            "remove_passage_ids": [],
            "confidence": 0.0,
            "reason": "Parsing failed",
        }


def detect_distractor(question, documents):
    evidence, id_map = format_evidence(documents)

    prompt = f"""
You are an evidence-conflict detector.

Use only the provided passages.

Question:
{question}

Passages:
{evidence}

Identify one passage that contains a clearly contradictory or
misleading answer-related claim.

Rules:
- Analyze all passages together.
- Preserve bridge and entity-linking evidence.
- Do not remove a passage only because it does not directly answer
  the question.
- If the conflict cannot be resolved, remove nothing.
- Select at most one passage.
- Use only neutral IDs such as P1 or P2.

Return only valid JSON:

{{
  "conflict_detected": true,
  "remove_passage_ids": ["P1"],
  "confidence": 0.0,
  "reason": "short explanation"
}}
"""

    response = judge_llm.invoke(prompt)
    decision = parse_response(response.content)

    decision["remove_passage_ids"] = [
        id_map[neutral_id]
        for neutral_id in decision["remove_passage_ids"]
        if neutral_id in id_map
    ]

    return decision


def repair_distractor(trace):
    repaired_trace = deepcopy(trace)

    documents = repaired_trace.get(
        "retrieval_events",
        [],
    )

    decision = detect_distractor(
        repaired_trace["question"],
        documents,
    )

    valid_ids = {str(document.get("passage_id", "")) for document in documents}

    requested_ids = [
        passage_id
        for passage_id in decision["remove_passage_ids"]
        if passage_id in valid_ids
    ]

    should_remove = (
        decision["conflict_detected"]
        and decision["confidence"] >= 0.70
        and len(requested_ids) == 1
    )

    remove_ids = set(requested_ids) if should_remove else set()

    filtered_documents = [
        document
        for document in documents
        if str(document.get("passage_id", "")) not in remove_ids
    ]

    repaired_trace["retrieval_events"] = filtered_documents

    history = list(repaired_trace.get("repair_history", []))

    history.append(
        {
            "failure": "D",
            "repair": "global_conflict_filtering",
            "decision": decision,
            "removed_passage_ids": list(remove_ids),
        }
    )

    repaired_trace["repair_history"] = history

    return repaired_trace
