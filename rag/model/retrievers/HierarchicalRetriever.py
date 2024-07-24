from wasabi import msg
from typing import Type, Callable

from rag.model.retrievers.base import BaseRAGRetriever, FilterUtil
from rag.type import *


class HierarchicalRetriever(BaseRAGRetriever):
    """Wrapper class that uses a retriever hierarchically.
    It retrieves base context first and then additional context.

    Args:
        retriever (BaseRAGRetriever): The retriever to use.
    """
    
    @classmethod
    def from_retriever(cls, retriever_builder: Callable[..., BaseRAGRetriever], **kwargs) -> "HierarchicalRetriever":
        retriever = retriever_builder(**kwargs)
        return cls(retriever)
        
    def __init__(
        self, 
        retriever: BaseRAGRetriever,
        **kwargs,
    ) -> None:
        super().__init__()
        self.retriever = retriever
        self.top_k = self.retriever.top_k
        self.base_ratio = 0.5
        
    def retrieve(self, queries: list[str], filter: Filter | None = None) -> list[Chunk]:
        # base context retrieval
        base_chunks = self.retriever.retrieve(
            queries, filter=FilterUtil.from_dict({"equals": {"key": "category", "value": "base"}})
        )
        # managed_base_chunks = self.rerank(base_chunks) # TODO topk
        managed_base_chunks = base_chunks[:int(self.top_k * self.base_ratio)]
        
        # additional context retrieval
        base_doc_ids = list(set([c.doc_id for c in managed_base_chunks]))
        additional_chunks = self.retriever.retrieve(
            queries, filter=FilterUtil.from_dict({
                "andAll": [
                    {"equals": {"key": "category", "value": "additional"}}, 
                    {"in": {"key": "base_doc_id", "value": base_doc_ids}}
                ]
            })
        )
        # managed_additional_chunks = self.rerank(additional_chunks) # TODO topk
        managed_additional_chunks = additional_chunks[:self.top_k - len(managed_base_chunks)]
        
        chunks = managed_base_chunks + managed_additional_chunks
        self.validate(chunks)
        
        return chunks
    
    def validate(self, chunks: list[Chunk]) -> None:
        """Validate the retrieved chunks.
            - category should be in ["base", "additional"]
            - base doc id of additional chunks should be in base chunks
        Args:
            chunks (list[Chunk]): The chunks to validate.
        """
        msg.info("Validating retrieved chunks...")
        # category should be in ["base", "additional"]
        try:
            for chunk in chunks:
                category = chunk.doc_meta.get("category")
                assert category in ["base", "additional"]
        except Exception as e:
            msg.warn(f"Validation failed: {e}")
        
        # base doc id of additional chunks should be in base chunks
        try:
            base_chunk_ids = set([
                chunk.doc_id for chunk in chunks if chunk.doc_meta.get("category") == "base"
            ])
            
            for chunk in chunks:
                if chunk.doc_meta.get("category") == "additional":
                    base_doc_id = chunk.doc_meta.get("base_doc_id")
                    assert base_doc_id in base_chunk_ids
        except Exception as e:
            msg.fail(f"Validation failed: {e}")
        
        msg.good("Validation passed.")
        return
    
    def rerank(self, chunks: list[Chunk]) -> list[Chunk]:
        # TODO
        # assume chunks are already sorted by score
        return chunks[:self.top_k]