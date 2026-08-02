from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


def create_vector_store(documents):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="comporepair_pilot",
        persist_directory="data/chroma_db",
    )

    return vector_store


def load_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma(
        collection_name="comporepair_pilot",
        embedding_function=embeddings,
        persist_directory="data/chroma_db",
    )

    return vector_store


def get_retriever(vector_store):
    return vector_store.as_retriever(search_kwargs={"k": 5})
