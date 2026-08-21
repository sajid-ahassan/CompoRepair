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
from copy import deepcopy

MAX_RETRIES = 1

BASE_SYSTEM_PROMPT = """
Role: You are a precise evidence-grounded question answering assistant.

Your task is to answer the given question using only the provided evidence.

Rules:
- Use only the provided evidence. Do not use outside knowledge.
- Answer the given question directly.
- Combine multiple passages when needed.
- Return ONLY the core final answer.
- Keep the answer as short as possible, usually 1-5 words.
- Do not give explanations, reasoning, introductions, or extra sentences.
- Do not say "the answer is", "based on the evidence", "I think", or similar phrases.
- Do not ask for clarification when a question is already provided.
- For yes/no questions, return only "yes" or "no".
- For a person, place, date, number, organization, or title, return only that value.
"""

SEMANTIC_JUDGE_PROMPT = """
Role: You are a semantic answer evaluator.

Your task is to decide whether the predicted answer gives the same core answer as the reference answer for the given question.

Rules:
- Allow capitalization and punctuation differences.
- Allow harmless wording differences.
- Allow abbreviations and equivalent short forms.
- Mark correct when the core meaning is clearly the same.
- Mark incorrect if the prediction does not answer the question.
- Mark incorrect if it gives a different person, place, date, number, entity, fact, or relation.

Return only the required verdict: correct or incorrect.
"""

llm = get_local_ollama_llm()

# =========================================================



class SemanticJudgeResult(BaseModel):
    verdict: Literal["correct", "incorrect"]


class ReasoningPlanDraft(BaseModel):
    evidence_task_1: str
    evidence_task_2: str
    final_task: str


# One shared model instance for generation and every structured-output task.


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_evidence(documents: Iterable[Any]) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        if hasattr(document, "page_content"):
            text = str(document.page_content).strip()
            title = str(document.metadata.get("title", "")).strip()
        else:
            text = str(document.get("text", "")).strip()
            title = str(document.get("title", "")).strip()

        header = f"[P{index}]"
        if title:
            header += f" Title: {title}"
        blocks.append(f"{header}\nText: {text}")
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
    input_tokens = int(usage.get("input_tokens") or metadata.get("prompt_eval_count") or 0)
    output_tokens = int(usage.get("output_tokens") or metadata.get("eval_count") or 0)
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
                f"Question:\n{question}\n\n"
                f"Evidence:\n{format_evidence(documents)}\n\nAnswer:"
            )
        ),
    ]
    result = invoke_text(messages)
    result["prompt_hash"] = hash_text(
        f"{system_prompt}\nQuestion:\n{{question}}\nEvidence:\n{{evidence}}"
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
        ]
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
        # Preserve deterministic EM/F1 and mark only the judge result as
        # indeterminate so one malformed structured response cannot stop the
        # complete experiment.
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