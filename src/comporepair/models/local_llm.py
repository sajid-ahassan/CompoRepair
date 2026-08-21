from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

def get_local_ollama_llm(
    model_name: str = "qwen3.5:9b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    reasoning: bool = False,
) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        validate_model_on_init=True,
        keep_alive="30m",
        reasoning=reasoning,
    )