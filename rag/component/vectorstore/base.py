from typing import Optional

from langchain_core.documents import Document

from rag.type import *

class BaseRAGVectorstore:
    def __init__(self, embeddings: Optional[Embeddings] = None, **kwargs) -> None:
        self.embeddings = embeddings
        self.vectorstore = None
        self._set_env()
        
    def _set_env(self):
        pass
    
    def ingest(self, chunks: list[Chunk]) -> int:
        raise NotImplementedError()
    
    def query(self, query: str, top_k: int = 5) -> list[Chunk]:
        raise NotImplementedError()
