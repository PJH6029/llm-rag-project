from wasabi import msg
from typing import Type, Callable

from rag.component.retriever.base import BaseRAGRetriever, FilterUtil
from rag.type import *


class HierarchicalRetriever(BaseRAGRetriever):
    """Wrapper class that uses a retriever hierarchically.
    It retrieves base context first and then additional context.

    Args:
        retriever (BaseRAGRetriever): The retriever to use.
    """
    BASE_RATIO = 0.7
    
    @classmethod
    def from_retriever(cls, retriever: BaseRAGRetriever) -> "HierarchicalRetriever":
        return cls(retriever)
        
    def __init__(
        self, 
        retriever: BaseRAGRetriever,
        **kwargs,
    ) -> None:
        super().__init__()
        self.retriever = retriever
        self.top_k = self.retriever.top_k
        
    def retrieve(self, queries: TransformationResult, filter: Filter | None = None) -> list[Chunk]:
        # base context retrieval
        base_filter = FilterUtil.from_dict({"equals": {"key": "doc_type", "value": "base"}})
        base_filter = FilterUtil.and_all(base_filter, filter) if filter else base_filter
        print(base_filter)
        base_chunks = self.retriever.retrieve(
            queries, filter=base_filter
        )
        managed_base_chunks = base_chunks[:int(self.top_k * self.BASE_RATIO)]
        
        # additional context retrieval 
        base_doc_ids = list(set([c.doc_id for c in managed_base_chunks])) + ["*"] # include additional docs not linked to any base doc
        additional_filters = FilterUtil.from_dict({
            "andAll": [
                {"equals": {"key": "doc_type", "value": "additional"}}, 
                {"in": {"key": "base_doc_id", "value": base_doc_ids}}
            ]
        })
        additional_filters = FilterUtil.and_all(additional_filters, filter) if filter else additional_filters
        print(additional_filters)
        additional_chunks = self.retriever.retrieve(
            queries, filter=additional_filters
        )
        managed_additional_chunks = additional_chunks[:self.top_k - len(managed_base_chunks)]
        
        chunks = managed_base_chunks + managed_additional_chunks
        self.validate(chunks)
        
        return chunks
    
    def validate(self, chunks: list[Chunk]) -> None:
        """Validate the retrieved chunks.
            - doc_type should be in ["base", "additional"]
            - base doc id of additional chunks should be in base chunks
        Args:
            chunks (list[Chunk]): The chunks to validate.
        """
        msg.info("Validating retrieved chunks...")
        # doc_type should be in ["base", "additional"]
        try:
            for chunk in chunks:
                doc_type = chunk.doc_meta.get("doc_type")
                assert doc_type in ["base", "additional"]
        except Exception as e:
            msg.fail(f"Validation failed: {e} doc_type should be in ['base', 'additional']")
            return
        
        # base doc id of additional chunks should be in base chunks
        try:
            base_chunk_ids = set([
                chunk.doc_id for chunk in chunks if chunk.doc_meta.get("doc_type") == "base"
            ] + ["*"])
            
            for chunk in chunks:
                if chunk.doc_meta.get("doc_type") == "additional":
                    base_doc_id = chunk.doc_meta.get("base_doc_id")
                    assert base_doc_id in base_chunk_ids
        except Exception as e:
            msg.fail(f"Validation failed: {e} Base doc id of additional chunks should be in base chunks")
            return
        
        msg.good("Validation passed.")
        return
