import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


from .run_baseline import normalize_answer, evaluate_answer

from .pipeline.baseline_rag import semantic_evaluation


from .repair.missing_evidence import repair_missing_evidence

from .repair.distractor import repair_distractor

from .repair.reasoning import repair_reasoning

from .repair.compound import repair_compound

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ==========================
# Answer Generation
# ==========================


def generate_answer(question, documents):

    context = "\n\n".join(doc["text"] for doc in documents)

    prompt = f"""

    Answer the question using only the provided evidence.

    Rules:
    - Return only the core answer.
    - Do not explain.
    - Do not provide reasoning.
    - Do not add extra words.

    Answer format:
    - For yes/no questions, return only yes or no.
    - For entities, names, locations, dates, or titles, return only that value.
    - For other questions, return only the core answer.
    Question:
    {question}

    Evidence:
    {context}

    Answer:
    """

    response = llm.invoke(prompt)

    return response.content.strip()


# ==========================
# Repair Selection
# ==========================


def select_repair(failures):

    if len(failures) > 1:
        return repair_compound

    failure = failures[0]

    if failure == "M":
        return repair_missing_evidence

    if failure == "D":
        return repair_distractor

    if failure == "G":
        return repair_reasoning

    raise ValueError(f"Unknown failure type: {failure}")


# ==========================
# Evaluation
# ==========================


def evaluate(question, answer, canonical_answer):

    return {
        "predicted_answer": normalize_answer(answer),
        "ground_truth": normalize_answer(canonical_answer),
        "exact_match": evaluate_answer(answer, canonical_answer),
        "semantic_correct": semantic_evaluation(question, canonical_answer, answer),
    }


# ==========================
# Main Repair Runner
# ==========================


def run_repair(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as f:

        traces = json.load(f)

    results = []

    for trace in traces[:2]:

        failures = trace.get("true_failures", [])

        repair_function = select_repair(failures)

        repaired_trace = repair_function(trace)

        # -------------------------
        # Generation after repair
        # -------------------------

        if "G" in failures and len(failures) == 1:

            # G repair already updates answer
            repaired_answer = repaired_trace.get("final_answer", "")

        else:

            repaired_answer = generate_answer(
                repaired_trace["question"], repaired_trace["retrieval_events"]
            )

        repaired_trace["repaired_answer"] = repaired_answer

        repaired_trace["repair_evaluation"] = {
            "before": {
                "answer": trace.get("final_answer", ""),
                "evaluation": trace.get("evaluation", {}),
            },
            "after": {
                "answer": repaired_answer,
                "evaluation": evaluate(
                    trace["question"], repaired_answer, trace["canonical_answer"]
                ),
            },
        }

        results.append(repaired_trace)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved repair results: {output_path}")


# ==========================
# Run All Repair Experiments
# ==========================

if __name__ == "__main__":

    experiments = [
        (
            "data/results/single/missing_evidence_results.json",
            "data/repaired_result/M_repair_results.json",
        ),
        (
            "data/results/single/distractor_results.json",
            "data/repaired_result/D_repair_results.json",
        ),
        (
            "data/results/single/reasoning_results.json",
            "data/repaired_result/G_repair_results.json",
        ),
        (
            "data/results/compound/M_D_results.json",
            "data/repaired_result/M_D_repair_results.json",
        ),
        (
            "data/results/compound/M_G_results.json",
            "data/repaired_result/M_G_repair_results.json",
        ),
        (
            "data/results/compound/D_G_results.json",
            "data/repaired_result/D_G_repair_results.json",
        ),
    ]

    for input_file, output_file in experiments:

        run_repair(input_file, output_file)
