from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def generate_node(state):

    context = "\n\n".join(state["evidence"])

    prompt = f"""

    Use only the evidence below.

    Evidence:
    {context}


    Question:
    {state["question"]}


    Answer:
    """

    response = llm.invoke(prompt)

    return {"answer": response.content}
