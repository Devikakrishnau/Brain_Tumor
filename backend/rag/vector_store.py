import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(RAG_DIR, "chroma_db")

# Use BioBERT or PubMedBERT embeddings
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

def get_vector_store():
    # Load ChromaDB from disk
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="medical_guidelines"
    )
    return vector_store

def add_documents_to_store(docs):
    vector_store = get_vector_store()
    vector_store.add_documents(docs)
    vector_store.persist()
    print(f"Added {len(docs)} documents to vector store.")

def query_guidelines(query: str, k: int = 3):
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=k)
    return results
