from src.comporepair.models.local_llm import get_local_ollama_llm
from typing import TypedDict
llm = get_local_ollama_llm()

prompt = f"""
say hello in one word
"""
    
resp = llm.invoke(prompt)
print(resp)