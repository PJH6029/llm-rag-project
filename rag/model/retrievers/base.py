from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.type import *

class BaseRAGRetriever:
    def __init__(self, top_k: int=5, embeddings: Optional[Embeddings] = None) -> None:
        self.top_k = top_k
        self.embeddings = embeddings
        self._set_env()
    
    def _set_env(self):
        """Set environment variables"""
        pass
    
    def retrieve(self, queries: list[str], filter: Optional[Filter]=None) -> list[Chunk]:
        raise []
    
    def _arange_filter(self, filter: Filter) -> dict:
        raise NotImplementedError()
    
    def process_chunk(self, chunk_raw: Document) -> Chunk:
        raise NotImplementedError()
    

class FilterUtil:
    @staticmethod
    def from_dict(filter: dict) -> Optional[Filter]:
        if not filter:
            return None
        if len(filter) > 2:
            raise ValueError("Invalid filter. Too many keys")
        
        op = list(filter.keys())[0]
        if op not in valid_ops:
            raise ValueError(f"Invalid operation: {op}")

        if op in filter_ops:
            try:
                operand = filter[op]
                key = operand["key"]
                value = operand["value"]
                predicate = FilterPredicate(op=op, key=key, value=value)
                return predicate
            except KeyError:
                raise ValueError("Invalid filter. Missing key or value")
        elif op in logical_ops:
            try:
                predicates = filter[op]
                predicates = [FilterUtil.from_dict(predicate) for predicate in predicates]
                expression = FilterExpression(op=op, predicates=predicates)
                return expression
            except KeyError:
                raise ValueError("Invalid filter. Missing predicates")
        else:
            # should never reach here
            raise ValueError("Invalid filter. Unknown operation")
