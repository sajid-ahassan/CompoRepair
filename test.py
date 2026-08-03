from src.comporepair.models.local_llm import get_local_ollama_llm

llm = get_local_ollama_llm()

prompt = f"""
Say hello in one word.
"""

resp = llm.invoke(prompt)
print(resp)