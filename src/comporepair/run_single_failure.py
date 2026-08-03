import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from .run_baseline import evaluate_answer, normalize_answer
from .pipeline.baseline_rag import semantic_evaluation

load_dotenv()

generation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ==========================
# Prompt Templates
# ==========================

BASE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """

Answer the question using only the provided evidence.

Rules:
- Do not use outside knowledge.
- Do not use your memory.
- Return only the core answer.
- Do not explain.
- Do not provide reasoning.
- Do not add extra words.

Answer format:
- For yes/no questions, return only yes or no.
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


G_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """

Answer the question using only the provided evidence.

Rules:
- Do not use outside knowledge.
- Prefer directly stated information.
- Avoid complex connections between separate passages.
- Return only the core answer.
- Do not explain.
- Do not provide reasoning.
- Do not add extra words.

Answer format:
- For yes/no questions, return only yes or no.
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


def select_prompt(failure_type):

    if failure_type == "G":
        return G_PROMPT

    return BASE_PROMPT


def generate_answer(question, documents, failure_type):

    context = "\n\n".join(doc["text"] for doc in documents)

    prompt = select_prompt(failure_type)
    

    messages = prompt.invoke({"question": question, "context": context})
    response = generation_llm.invoke(messages)

    return response.content.strip()


def run_single_failure(input_path, output_path, failure_type):

    with open(input_path, "r", encoding="utf-8") as f:

        traces = json.load(f)

    results = []

    for trace in traces:

        answer = generate_answer(
            trace["question"], trace["retrieval_events"], failure_type
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

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":

    run_single_failure(
        "data/failures/single/missing_evidence.json",
        "data/results/single/missing_evidence_results.json",
        "M",
    )

    run_single_failure(
        "data/failures/single/distractor.json",
        "data/results/single/distractor_results.json",
        "D",
    )

    run_single_failure(
        "data/failures/single/reasoning.json",
        "data/results/single/reasoning_results.json",
        "G",
    )