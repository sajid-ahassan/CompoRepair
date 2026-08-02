from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from .state import RAGState
from ..retrieval.vector_store import load_vector_store, get_retriever


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
from dotenv import load_dotenv
load_dotenv()





def retrieval_node(state: RAGState, retriever):

    docs = retriever.invoke(state["question"])

    context = "\n\n".join(doc.page_content for doc in docs)

    return {"retrieved_documents": docs, "context": context}


def generation_node(state: RAGState):


    prompt = f"""
    Answer the question using the provided context.

    Question:
    {state["question"]}

    Context:
    {state["context"]}

    Answer:
    """

    response = llm.invoke(prompt)

    return {"answer": response.content}


def build_baseline_graph(retriever):

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", lambda state: retrieval_node(state, retriever))

    graph.add_node("generate", generation_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "generate")

    graph.add_edge("generate", END)

    return graph.compile()




def run_baseline(question: str):

    vector_store = load_vector_store()

    retriever = get_retriever(vector_store)

    graph = build_baseline_graph(retriever)

    result = graph.invoke(
        {
            "question": question,
            "retrieved_documents": [],
            "context": "",
            "answer": ""
        }
    )

    return result