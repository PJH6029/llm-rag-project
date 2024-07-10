from rag.interfaces import Embedder, Retriever
from rag.types import *

from langchain_community.vectorstores import FAISS

from wasabi import msg


class FAISSRetriever(Retriever):
    def __init__(self) -> None:
        super().__init__()
        self.top_k = 5
        # TODO embedder
        self.faiss_vectorstore: FAISS = FAISS.from_documents()
    
    def retrieve(self, queries: list[str], embedder: Embedder) -> tuple[list[Chunk], str]:
        faiss_retriever = self.faiss_vectorstore.as_retriever(search_kwargs={'k': self.top_k})
        msg.info("FAISS Retriever initialized. Retrieving chunks...")
        chunks = []
        for query in queries:
            retrieved_chunks = faiss_retriever.invoke(query)
            for chunk_raw in retrieved_chunks:
                # TODO
                pass