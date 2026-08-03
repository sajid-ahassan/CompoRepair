import json

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


reasoning_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def verify_answer(question, evidence, answer):

    prompt = f"""

    Check whether the answer is supported by the evidence.

    Question:
    {question}


    Evidence:
    {evidence}


    Generated Answer:
    {answer}


    Strictly Return ONLY valid JSON:

    {{
        "supported": true,
        "corrected_answer": "",
        "reason": ""
    }}


    Strictly Required Rules:

    - supported=true if the answer is directly supported by evidence.
    - supported=false if the answer is unsupported or incorrect.
    - If unsupported, provide the best answer using only the evidence.
            Answer format:
            - For yes/no questions, return only: yes or no.
            - For names, locations, dates, organizations, or titles, return only that value.
            - For other questions, return only the core answer.
    
    """

    response = reasoning_llm.invoke(prompt)

    return parse_verification(response.content)


def parse_verification(response):

    try:

        result = json.loads(response)

        return {
            "supported": result.get("supported", False),
            "corrected_answer": result.get("corrected_answer", ""),
            "reason": result.get("reason", ""),
        }

    except Exception:

        return {"supported": False, "corrected_answer": "", "reason": "Parsing failed"}


def repair_reasoning(trace):

    repaired_trace = trace.copy()

    context = "\n\n".join(doc["text"] for doc in trace["retrieval_events"])

    # Existing answer from failed run

    old_answer = trace.get("final_answer", "")

    verification = verify_answer(trace["question"], context, old_answer)

    if verification["supported"]:

        repaired_answer = old_answer

    else:

        repaired_answer = verification["corrected_answer"]

    repaired_trace["final_answer"] = repaired_answer

    history = list(repaired_trace.get("repair_history", []))
    history.append(
        {
            "failure": "G",
            "repair": "answer_verification",
            "verification": verification,
        }
    )

    repaired_trace["repair_history"] = history

    return repaired_trace
