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

    Answer the question using ONLY the information explicitly stated in the context.

    Strict rules:
    - Do not use outside knowledge.
    - Do not use your memory.
    - Do not infer missing information.
    - If the answer is not explicitly present in the context, return exactly:
    unknown

    Output rules:
    - Return only the answer.
    - Do not explain.
    - Do not provide reasoning.
    - Do not add extra words.

    Question:
    {question}

    Context:
    {context}

    Final answer:
    """
    response = generation_llm.invoke(prompt)

    return response.content.strip()


def main():

    with open("data/failures/single/missing_evidence.json", "r", encoding="utf-8") as f:
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

    with open("data/results/missing_evidence_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    


if __name__ == "__main__":
    main()
