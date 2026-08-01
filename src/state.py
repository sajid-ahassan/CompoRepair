from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph


class RAGState(StateGraph):

    question: str

    retrieved_documents: List[str]

    evidence: List[str]

    answer: str

    verification: Dict[str, Any]

    trace: Dict[str, Any]
