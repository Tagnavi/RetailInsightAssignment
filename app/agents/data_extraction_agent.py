from typing import List
from langchain_core.documents import Document
from app.retrieval.rag_store import RetailRAGStore


class DataExtractionAgent:
    """
    Data extraction agent for RAG: retrieves relevant documents
    from the vector store using the normalized query.
    """

    def __init__(self, folder_path: str):
        self.store = RetailRAGStore()
        self.store.ingest_folder(folder_path)
        self.retriever = self.store.as_retriever(k=6)

    def retrieve_context(self, query: str) -> List[Document]:
        # New retriever API
        return self.retriever.invoke(query)
