import json
from langchain_core.documents import Document


def load_trace_documents(path: str):

    with open(path, "r", encoding="utf-8") as f:
        traces = json.load(f)

    documents = []

    for trace in traces:
        for evidence in trace["selected_evidence"]:

            documents.append(
                Document(
                    page_content=evidence["text"],
                    metadata={
                        "question_id": trace["question_id"],
                        "passage_id": evidence["passage_id"],
                        "title": evidence["title"],
                        "is_supporting": evidence["is_supporting"],
                        "document_role": evidence["document_role"],
                        "rank": evidence["rank"],
                    },
                )
            )

    return documents


# documents = load_trace_documents(
#     "data/processed/pilot_base_traces.json"
# )
