import json
import os

from langchain_core.prompts import ChatPromptTemplate
from .run_baseline import evaluate_answer, normalize_answer
from .pipeline.baseline_rag import semantic_evaluation, generation_llm

generation_llm = generation_llm

STANDARD_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Answer the question using only the provided evidence.

Rules:
- Return only the final answer.
- Return the shortest possible answer.
- Do not explain.
- Do not provide reasoning.
- Do not write complete sentences.
- Do not add phrases like "The answer is".

Answer format:
- For yes/no questions, return only: yes or no.
- For names, locations, dates, organizations, or titles, return only that value.
- For other questions, return only the core answer.
""",
        ),
        (
            "human",
            """
Question:
{question}

Evidence:
{context}

Answer:
""",
        ),
    ]
)

G_FAILURE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """

Answer the question using only the provided evidence.

Rules:
- Do not use outside knowledge.
- Do not use your memory.
- Prefer directly stated information.
- Avoid complex connections between separate passages.
- Return only the core answer.
- Do not explain.
- Do not provide reasoning.
- Do not add extra words.


Answer format:
- For yes/no questions, return only: yes or no.
- For names, locations, dates, organizations, or titles, return only that value.
- For other questions, return only the core answer.
""",
        ),
        (
            "human",
            """
Question:
{question}

Evidence:
{context}

Answer:
""",
        ),
    ]
)


def select_prompt(failures: list[str]) -> ChatPromptTemplate:
    if "G" in failures:
        return G_FAILURE_PROMPT

    return STANDARD_PROMPT


def generate_from_compound_context(
    question: str,
    documents: list[dict],
    failures: list[str],
) -> str:

    context = "\n\n".join(document["text"] for document in documents)

    prompt_template = select_prompt(failures)

    messages = prompt_template.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    response = generation_llm.invoke(messages)

    return response.content.strip()


def run_compound_failure(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as f:
        traces = json.load(f)

    results = []

    for trace in traces[:5]:

        answer = generate_from_compound_context(
            trace["question"], trace["retrieval_events"], trace["true_failures"]
        )

        trace["final_answer"] = answer

        trace["answer_claims"] = [{"claim": answer, "source": "llm_generation"}]

        trace["evaluation"] = {
            "predicted_answer": normalize_answer(answer),
            "ground_truth": normalize_answer(trace["canonical_answer"]),
            "exact_match": evaluate_answer(answer, trace["canonical_answer"]),
            "semantic_correct": semantic_evaluation(
                trace["question"], trace["canonical_answer"], answer
            ),
        }

        results.append(trace)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Compound results saved: {output_path}")


if __name__ == "__main__":

    run_compound_failure(
        "data/failures/compound/M_D.json", "data/results/compound/M_D_results.json"
    )

    run_compound_failure(
        "data/failures/compound/M_G.json", "data/results/compound/M_G_results.json"
    )

    run_compound_failure(
        "data/failures/compound/D_G.json", "data/results/compound/D_G_results.json"
    )
