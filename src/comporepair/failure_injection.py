import json
import os
import random
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==========================
# Missing Evidence (M)
# ==========================


def inject_missing_evidence(trace):

    corrupted_trace = trace.copy()

    retrieved_docs = trace["retrieval_events"]

    supporting_ids = {
        doc["passage_id"] for doc in trace["selected_evidence"] if doc["is_supporting"]
    }

    corrupted_retrieval = [
        doc for doc in retrieved_docs if doc["passage_id"] not in supporting_ids
    ]

    corrupted_trace["retrieval_events"] = corrupted_retrieval

    corrupted_trace["true_failures"] = ["M"]

    corrupted_trace["experiment_stage"] = "single_missing_evidence"

    return corrupted_trace


def create_missing_evidence_dataset(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as f:
        traces = json.load(f)

    corrupted = []

    for trace in traces:
        corrupted.append(inject_missing_evidence(trace))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corrupted, f, indent=2, ensure_ascii=False)

    print(f"Missing evidence dataset saved: {output_path}")


# ==========================
# Distractor Evidence (D)
# ==========================


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def generate_distractor(question, supporting_text):

    prompt = f"""
    You are creating a contradictory distractor passage for a RAG evaluation dataset.

    Given the question and the correct evidence, generate a misleading passage.

    Roles:
    - Keep the same person/entity.
    - Keep the same topic and surrounding facts.
    - Modify only the key fact required to answer the question.
    - Do not introduce unrelated positions or events.
    - Make the false information plausible.
    - Do not add extra background information.
    - Do not explain that it is a distractor.
    - Write it like a real retrieved document.
    - The distractor should look like a near-correct passage, not a different biography.
    
    Question:
    {question}

    Correct evidence:
    {supporting_text}

    Contradictory distractor passage:
    """

    response = llm.invoke(prompt)

    return response.content.strip()


def inject_distractor_evidence(trace):

    corrupted_trace = trace.copy()

    corrupted_retrieval = trace["retrieval_events"].copy()

    # Select one supporting evidence
    supporting_docs = [
        doc for doc in trace["selected_evidence"] if doc["is_supporting"]
    ]

    if supporting_docs:

        source_doc = supporting_docs[0]

        distractor_text = generate_distractor(trace["question"], source_doc["text"])

        corrupted_retrieval.append(
            {
                "title": source_doc["title"],
                "passage_id": source_doc["passage_id"] + "_distractor",
                "rank": len(corrupted_retrieval) + 1,
                "text": distractor_text,
                "document_role": "distractor",
            }
        )

    random.shuffle(corrupted_retrieval)
    for index, doc in enumerate(corrupted_retrieval):
        doc["rank"] = index + 1

    corrupted_trace["retrieval_events"] = corrupted_retrieval

    corrupted_trace["true_failures"] = ["D"]

    corrupted_trace["experiment_stage"] = "single_distractor_evidence"

    return corrupted_trace


def create_distractor_dataset(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as f:
        traces = json.load(f)

    corrupted = []

    for trace in traces:
        corrupted.append(inject_distractor_evidence(trace))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corrupted, f, indent=2, ensure_ascii=False)

    print(f"Distractor dataset saved: {output_path}")


# ==========================
# Reasoning Failure (G)
# =========================


def inject_reasoning_failure(trace):

    corrupted_trace = trace.copy()

    # Evidence remains unchanged
    corrupted_trace["retrieval_events"] = trace["retrieval_events"]

    corrupted_trace["true_failures"] = ["G"]

    corrupted_trace["experiment_stage"] = "single_reasoning_failure"

    corrupted_trace["reasoning_constraint"] = "Do not perform multi-hop reasoning"

    return corrupted_trace


def create_reasoning_dataset(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as f:
        traces = json.load(f)

    corrupted = []

    for trace in traces:
        if trace["dataset_metadata"]["type"] in ["bridge", "comparison"]:
            corrupted.append(inject_reasoning_failure(trace))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corrupted, f, indent=2, ensure_ascii=False)

    print(f"Reasoning dataset saved: {output_path}")


# ==========================
# Compound Failures
# ==========================



def inject_compound_failure(trace, failures):

    corrupted_trace = trace.copy()

    # Apply failures sequentially

    if "M" in failures:
        corrupted_trace = inject_missing_evidence(corrupted_trace)

    if "D" in failures:
        corrupted_trace = inject_distractor_evidence(corrupted_trace)

    if "G" in failures:
        corrupted_trace = inject_reasoning_failure(corrupted_trace)

    corrupted_trace["true_failures"] = failures

    corrupted_trace["experiment_stage"] = "compound_" + "_".join(failures)

    return corrupted_trace


def create_compound_dataset(input_path, output_path, failures):

    with open(input_path, "r", encoding="utf-8") as f:
        traces = json.load(f)

    corrupted = []

    for trace in traces:
        corrupted.append(inject_compound_failure(trace, failures))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corrupted, f, indent=2, ensure_ascii=False)

    print(f"Compound dataset saved: {output_path}")


# ==========================
# Main
# ==========================

if __name__ == "__main__":

    input_file = "data/processed/pilot_baseline_results.json"

    # create_missing_evidence_dataset(
    #     input_file, "data/failures/single/missing_evidence.json"
    # )

    # create_distractor_dataset(input_file, "data/failures/single/distractor.json")

    # create_reasoning_dataset(input_file, "data/failures/single/reasoning.json")
    
    # for compound failures
    
    create_compound_dataset(
        input_file,
        "data/failures/compound/M_D.json",
        ["M", "D"]
    )


    create_compound_dataset(
        input_file,
        "data/failures/compound/M_G.json",
        ["M", "G"]
    )


    create_compound_dataset(
        input_file,
        "data/failures/compound/D_G.json",
        ["D", "G"]
    )
