from langgraph.graph import StateGraph, END,START

from .state import RAGState

from .retriever import retrieve_node
from .generator import generate_node
from .verifier import verify_node


def build_graph():

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()
