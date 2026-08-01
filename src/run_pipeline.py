import json
from .graph import build_graph
from .retriever import initialize_vectorstore
from langchain_core.documents import Document

with open("data/pilot_questions.json", encoding="utf-8") as f:
    questions = json.load(f)
    
# print(questions[0]['context']['title'])
documents = []

for item in questions:
    titles = item["context"]["title"]
    sentence_groups = item["context"]["sentences"]

    for title, sentences_in_doc in zip(titles, sentence_groups):
        # Combine list of sentences into a single block of text
        full_text = " ".join(sentences_in_doc)
        
        documents.append(
            Document(
                page_content=full_text,
                metadata={
                    "title": title,
                    "id": item["id"]
                }
            )
        )

print(documents[0].page_content, documents[0].metadata)

# initialize_vectorstore(documents)
# app = build_graph()

# for item in questions[:1]:

#     result = app.invoke({"question": item["question"]})

#     print(result["trace"])
