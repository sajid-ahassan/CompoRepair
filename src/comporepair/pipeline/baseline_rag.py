import hashlib
import re
import string
import time
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, Iterable, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from .state import RAGState
from ..models.local_llm import get_local_ollama_llm
from ..retrieval.vector_store import RETRIEVAL_K, get_retriever

MAX_RETRIES = 1

BASE_SYSTEM_PROMPT = """Answer the question using only the provided evidence.

Strict Rules:
- Do not use outside knowledge or memory.
- To get the final answer, you may need to combine information from multiple passages.
- You may need to reason across multiple passages to get the final answer.
- Combine information across given passages when needed.
- Return only the concise final answer.
- Do not explain or show reasoning.
- Do not include explanations, comparisons, dates, reasons, evidence, or extra words.
- If the question asks "Who...", return only the person's name.
- If the question asks "Which film...", return only the film title.
- If the question asks "Where...", return only the place.
- If the question asks "When..." or asks for a date, return only the date/year.
- If the question asks which of two candidates is younger, older, earlier, later, larger, smaller, etc., return only the selected candidate's name.
- Never restate the question.
- Never write "X is younger than Y", "X because...", or similar explanatory text.
- Never ask the user to provide the question.

Return only the concise final answer, normally 1-5 words.
"""


SEMANTIC_JUDGE_PROMPT = """
Role: You are a strict semantic answer evaluator.

Determine whether the predicted answer directly gives the SAME answer
as the reference answer for the given question.

Important:
- The reference answer is authoritative.
- Judge the answer to the QUESTION, not whether the prediction is related
  to the same topic.
- A related person, film, place, date, country, or intermediate entity is
  NOT the same answer.
- For comparison questions, the prediction must select the same candidate
  as the reference.
- If the question asks for a film, returning its director is incorrect.
- If the question asks for a person, returning that person's parent,
  child, spouse, or another related person is incorrect.
- If the question asks for a place, returning a person is incorrect.
- If the question asks for a date/year, a refusal or missing-information
  statement is incorrect.
- A prediction that says the answer cannot be determined, is unavailable,
  or is not in the evidence is incorrect when the reference contains an answer.
- Do not mark an answer correct merely because words from the reference
  appear somewhere in the prediction.
- Extra wording is allowed when the prediction clearly contains the
  same final answer as the reference and does not introduce a conflicting
  alternative answer.
- Allow capitalization, punctuation, spelling variants, common
  abbreviations, and genuinely equivalent names.

Return only: correct or incorrect.

Some examples:
Question: Which film has the younger director?
Reference: The World of Apu
Prediction: Satyajit Ray
Verdict: incorrect

Question: What is the date of death of X's husband?
Reference: 548
Prediction: There is no information about X's husband.
Verdict: incorrect

Question: What nationality is X's husband?
Reference: Venezuelan
Prediction: X's husband is Venezuelan.
Verdict: correct

Question: Who is X's maternal grandfather?
Reference: Sextus Aelius Catus
Prediction: X's maternal grandfather is Sextus Aelius Catus.
Verdict: correct

Reference: 24 March 1927
Prediction: Elizabeth Mavrikievna died on 24 March 1927.
→ correct
"""

# One shared model instance for generation and every structured-output task.
llm = get_local_ollama_llm()


class SemanticJudgeResult(BaseModel):
    verdict: Literal["correct", "incorrect"]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_evidence(documents: Iterable[Any]) -> str:
    """Format evidence without exposing passage titles to any LLM."""
    blocks = []
    for index, document in enumerate(documents, start=1):
        if hasattr(document, "page_content"):
            text = str(document.page_content).strip()
        else:
            text = str(document.get("text", "")).strip()
        blocks.append(f"[P{index}]\nText: {text}")
    return "\n\n".join(blocks)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return ""


def _token_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    input_tokens = int(
        usage.get("input_tokens") or metadata.get("prompt_eval_count") or 0
    )
    output_tokens = int(
        usage.get("output_tokens") or metadata.get("eval_count") or 0
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def invoke_text(messages: Any) -> Dict[str, Any]:
    """Generate text, retrying once only when the response is empty or raises."""
    start = time.perf_counter()
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = llm.invoke(messages)
            usage = _token_usage(response)
            text = _response_text(response)
            if text:
                return {
                    "text": text,
                    "generation_failed": False,
                    "attempts": attempt,
                    "latency_ms": int((time.perf_counter() - start) * 1000),
                    "token_usage": usage,
                }
        except Exception:
            pass

    return {
        "text": "",
        "generation_failed": True,
        "attempts": MAX_RETRIES + 1,
        "latency_ms": int((time.perf_counter() - start) * 1000),
        "token_usage": usage,
    }


def invoke_structured(runnable: Any, payload: Any) -> Any:
    """Invoke a structured-output runnable with the shared retry policy."""
    last_error = None
    for _ in range(MAX_RETRIES + 1):
        try:
            return runnable.invoke(payload)
        except Exception as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError("Structured invocation failed without an exception.")


def generate_answer(
    question: str,
    documents: Iterable[Any],
    additional_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    system_prompt = BASE_SYSTEM_PROMPT
    if additional_instruction:
        system_prompt += f"\n{additional_instruction.strip()}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"The question you need to answer is:\n{question}\n\n"
                f"Passages:\n{format_evidence(documents)}\n\nAnswer:"
            )
        ),
    ]
    result = invoke_text(messages)
    result["prompt_hash"] = hash_text(
        f"{system_prompt}\nQuestion:\n{{question}}\nPassages:\n{{evidence}}"
    )
    return result


def semantic_evaluation(
    question: str,
    reference_answer: str,
    predicted_answer: str,
) -> Dict[str, bool]:
    judge = llm.with_structured_output(
        SemanticJudgeResult,
        method="json_schema",
    )
    result = invoke_structured(
        judge,
        [
            SystemMessage(content=SEMANTIC_JUDGE_PROMPT),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    f"Reference answer:\n{reference_answer}\n\n"
                    f"Predicted answer:\n{predicted_answer}"
                )
            ),
        ],
    )
    return {
        "semantic_correct": result.verdict == "correct",
        "semantic_judge_error": False,
    }


def normalize_answer(answer: Optional[str]) -> str:
    text = "" if answer is None else str(answer).lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: Optional[str], reference: Optional[str]) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def token_f1(prediction: Optional[str], reference: Optional[str]) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_prediction(
    question: str,
    prediction: Optional[str],
    reference_answer: str,
    generation_failed: bool = False,
) -> Dict[str, Any]:
    if generation_failed or not str(prediction or "").strip():
        return {
            "predicted_answer": "",
            "canonical_answer": normalize_answer(reference_answer),
            "exact_match": False,
            "token_f1": 0.0,
            "semantic_correct": False,
            "generation_failed": True,
            "semantic_judge_error": False,
        }

    try:
        semantic = semantic_evaluation(
            question,
            reference_answer,
            str(prediction),
        )
        semantic_correct = bool(semantic["semantic_correct"])
        semantic_judge_error = False
    except Exception:
        semantic_correct = False
        semantic_judge_error = True

    return {
        "predicted_answer": normalize_answer(prediction),
        "canonical_answer": normalize_answer(reference_answer),
        "exact_match": exact_match(prediction, reference_answer),
        "token_f1": token_f1(prediction, reference_answer),
        "semantic_correct": semantic_correct,
        "generation_failed": False,
        "semantic_judge_error": semantic_judge_error,
    }


def model_manifest() -> Dict[str, Any]:
    return {
        "provider": "ollama",
        "model": str(getattr(llm, "model", "")),
        "retrieval_k": RETRIEVAL_K,
    }


def retrieval_node(state: RAGState, retriever):
    documents = retriever.invoke(state["question"])
    return {"retrieved_documents": documents, "context": format_evidence(documents)}


def generation_node(state: RAGState):
    result = generate_answer(state["question"], state["retrieved_documents"])
    return {
        "answer": result["text"],
        "generation_failed": result["generation_failed"],
        "generation_attempts": result["attempts"],
        "latency_ms": result["latency_ms"],
        "token_usage": result["token_usage"],
        "prompt_hash": result["prompt_hash"],
    }


@lru_cache(maxsize=1)
def build_baseline_graph():
    retriever = get_retriever(k=RETRIEVAL_K)
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", lambda state: retrieval_node(state, retriever))
    graph.add_node("generate", generation_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_baseline(question: str, reference_answer: str) -> Dict[str, Any]:
    result = build_baseline_graph().invoke({"question": question})
    result["evaluation"] = evaluate_prediction(
        question,
        result.get("answer", ""),
        reference_answer,
        result.get("generation_failed", False),
    )
    return result
