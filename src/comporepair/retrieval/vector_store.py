import os
import shutil
from functools import lru_cache

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

load_dotenv()

COLLECTION_NAME = "comporepair_pilot"
PERSIST_DIRECTORY = "data/chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_K = 12
REPAIR_CANDIDATE_K = 20


def _embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
    # return OllamaEmbeddings(model=EMBEDDING_MODEL,dimensions=2560)


def create_vector_store(documents):
    if os.path.isdir(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    ids = [str(document.metadata["passage_id"]) for document in documents]
    vector_store = Chroma.from_documents(
        documents=documents,
        ids=ids,
        embedding=_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )
    load_vector_store.cache_clear()
    return vector_store


@lru_cache(maxsize=1)
def load_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )


def get_retriever(vector_store=None, k=RETRIEVAL_K):
    store = vector_store or load_vector_store()
    return store.as_retriever(search_kwargs={"k": int(k)})
