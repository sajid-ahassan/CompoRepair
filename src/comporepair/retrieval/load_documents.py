import json

from langchain_core.documents import Document


def load_trace_documents(path: str):
    with open(path, "r", encoding="utf-8") as file:
        traces = json.load(file)

    documents = []
    for trace in traces:
        for evidence in trace.get("selected_evidence", []):
            documents.append(
                Document(
                    page_content=evidence["text"],
                    metadata={
                        "question_id": trace["question_id"],
                        "passage_id": evidence["passage_id"],
                        "title": evidence["title"],
                        "is_supporting": bool(evidence.get("is_supporting", False)),
                        "document_role": evidence.get(
                            "document_role", "non_supporting"
                        ),
                        "rank": int(evidence.get("rank", 0)),
                    },
                )
            )

    return documents
