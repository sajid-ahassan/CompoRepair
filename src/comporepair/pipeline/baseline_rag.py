from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from .state import RAGState
from ..retrieval.vector_store import load_vector_store, get_retriever

from dotenv import load_dotenv

load_dotenv()


generation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

evaluation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def retrieval_node(state: RAGState, retriever):

    docs = retriever.invoke(state["question"])

    context = "\n\n".join(doc.page_content for doc in docs)

    return {"retrieved_documents": docs, "context": context}


def generation_node(state: RAGState):

    prompt = f"""
Answer the question using only the provided context.

Rules:
- Return only the answer.
- Do not explain.
- Do not provide reasoning.
- Do not write a complete sentence.
- Keep the answer as short as possible.

Question:
{state["question"]}

Context:
{state["context"]}

Answer:
"""

    response = generation_llm.invoke(prompt)

    return {"answer": response.content.strip()}


def semantic_evaluation(question, reference_answer, predicted_answer):

    prompt = f"""
You are evaluating a question answering system.

Determine whether the predicted answer is semantically correct compared to the reference answer.

Consider:
- Different word order is acceptable.
- Different capitalization is acceptable.
- Shorter answers are acceptable only if they contain the complete required meaning.
- Do not require exact wording.

Question:
{question}

Reference answer:
{reference_answer}

Predicted answer:
{predicted_answer}

Return only one word:
correct
or
incorrect
"""

    response = evaluation_llm.invoke(prompt)

    result = response.content.strip().lower()

    return result == "correct"


def build_baseline_graph(retriever):

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", lambda state: retrieval_node(state, retriever))

    graph.add_node("generate", generation_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "generate")

    graph.add_edge("generate", END)

    return graph.compile()


def run_baseline(question: str, reference_answer: str):

    vector_store = load_vector_store()

    retriever = get_retriever(vector_store)

    graph = build_baseline_graph(retriever)

    result = graph.invoke(
        {"question": question, "retrieved_documents": [], "context": "", "answer": ""}
    )

    predicted_answer = result["answer"]

    semantic_score = semantic_evaluation(question, reference_answer, predicted_answer)

    result["evaluation"] = {
        "reference_answer": reference_answer,
        "predicted_answer": predicted_answer,
        "semantic_correct": semantic_score,
    }

    return result
