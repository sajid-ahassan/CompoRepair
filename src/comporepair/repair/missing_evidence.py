from copy import deepcopy
from functools import lru_cache

from ..retrieval.vector_store import (
    load_vector_store,
    get_retriever,
)


@lru_cache(maxsize=1)
def load_repair_retriever():

    vector_store = load_vector_store()

    return get_retriever(vector_store)


def repair_missing_evidence(trace: dict) -> dict:
    repaired_trace = deepcopy(trace)

    retriever = load_repair_retriever()

    docs = retriever.invoke(repaired_trace["question"])

    recovered_documents = []

    for index, doc in enumerate(docs, start=1):

        passage_id = doc.metadata.get("passage_id")

        if not passage_id:
            continue

        recovered_documents.append(
            {
                "title": doc.metadata.get(
                    "title",
                    "",
                ),
                "passage_id": passage_id,
                "rank": index,
                "text": doc.page_content,
                "document_role": "repair_retrieval",
            }
        )

    existing_documents = repaired_trace.get(
        "retrieval_events",
        [],
    )

    merged_documents = {}

    for document in recovered_documents:

        passage_id = document.get("passage_id")

        if passage_id:
            merged_documents[passage_id] = document

    # Preserve existing unique evidence, including distractors.
    for document in existing_documents:

        passage_id = document.get("passage_id")

        if passage_id and passage_id not in merged_documents:
            merged_documents[passage_id] = document

    final_documents = list(merged_documents.values())

    # Reset ranks after merging.
    for index, document in enumerate(
        final_documents,
        start=1,
    ):
        document["rank"] = index

    repaired_trace["retrieval_events"] = final_documents

    repair_history = list(
        repaired_trace.get(
            "repair_history",
            [],
        )
    )

    repair_history.append(
        {
            "failure": "M",
            "repair": "retrieval_recovery",
            "documents_before": len(existing_documents),
            "retrieved_count": len(recovered_documents),
            "documents_after": len(final_documents),
        }
    )

    repaired_trace["repair_history"] = repair_history

    return repaired_trace
