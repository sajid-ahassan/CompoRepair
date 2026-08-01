from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


vectorstore = None

def initialize_vectorstore(documents):

    global vectorstore

    vectorstore = Chroma.from_documents(documents, embedding_model,persist_directory="data/chroma_db")

def retrieve_node(state):

    docs = vectorstore.similarity_search(state["question"], k=5)

    contents = [d.page_content for d in docs]

    return {"retrieved_documents": contents, "evidence": contents}
