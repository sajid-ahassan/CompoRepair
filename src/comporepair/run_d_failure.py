import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from .run_baseline import evaluate_answer, normalize_answer
from .pipeline.baseline_rag import semantic_evaluation

load_dotenv()

generation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def generate_from_failure_context(question, documents):

    context = "\n\n".join(doc["text"] for doc in documents)
    prompt = f"""
    You are a question answering system.

    Answer the question using the provided evidence.

    Rules:
    - Prefer evidence directly relevant to the question.
    - Do not use outside knowledge.
    - Return only the final answer.
    - Do not explain.

    Question:
    {question}

    Evidence:
    {context}

    Final answer:
    """
    print(f"Prompt for question '{question}':\n{prompt}\n")
    response = generation_llm.invoke(prompt)

    return response.content.strip()


def main():

    with open("data/failures/single/distractor.json", "r", encoding="utf-8") as f:
        traces = json.load(f)

    results = []

    for trace in traces[:1]:

        answer = generate_from_failure_context(
            trace["question"], trace["retrieval_events"]
        )

        trace["final_answer"] = answer
        trace["answer_claims"] = [
            {
                "claim": answer,
                "source": "llm_generation"
            }
        ]

        trace["evaluation"] = {
            "predicted_answer": normalize_answer(answer),
            "ground_truth": normalize_answer(trace["canonical_answer"]),
            "exact_match": evaluate_answer(
                answer,
                trace["canonical_answer"]
            ),
            "semantic_correct": semantic_evaluation(
                trace["question"],
                trace["canonical_answer"],
                answer
            )
        }

        results.append(trace)

    with open("data/results/distractor_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Results saved to data/results/distractor_results.json")


if __name__ == "__main__":
    main()
