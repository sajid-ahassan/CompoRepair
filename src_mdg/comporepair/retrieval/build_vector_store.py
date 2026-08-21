from .load_documents import load_trace_documents
from .vector_store import create_vector_store


def main():

    documents = load_trace_documents("data/processed/pilot_base_traces.json")

    create_vector_store(documents)

    print("Chroma database created successfully.")


if __name__ == "__main__":
    main()
