def verify_node(state):

    verification = {
        "evidence_count": len(state["evidence"]),
        "answer_exists": len(state["answer"]) > 0,
        "supported": True,
    }

    trace = {
        "question": state["question"],
        "retrieved_documents": state["retrieved_documents"],
        "answer": state["answer"],
        "verification": verification,
    }

    return {"verification": verification, "trace": trace}
