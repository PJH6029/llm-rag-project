from wasabi import msg
import os
from typing import Optional, Iterator, Iterable, Callable, Any, TypeVar, Hashable
from itertools import chain
from collections import defaultdict

from rag.component.retriever.base import BaseRAGRetriever
from rag.type import *
from rag.util import generate_id

T = TypeVar("T")
H = TypeVar("H", bound=Hashable)

def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e

class EnsembleRetriever(BaseRAGRetriever):
    def __init__(
        self, 
        retrievers: list[BaseRAGRetriever], 
        weights: Optional[list[float]]=None,
        top_k: int=5,
        **kwargs
    ) -> None:
        super().__init__()
        self.retrievers = retrievers
        self.weights = weights
        self.c = 60
        self.top_k = top_k
    
    def retrieve(self, queries: TransformationResult, filter: Filter | None = None) -> list[Chunk]:
        retrieved_chunks_list = [
            retriever.retrieve(queries, filter) for retriever in self.retrievers
        ]
        # invocation_cnt = len(self.retrievers) * len(queries) TODO trace invocation count
        
        return self.weighted_reciprocal_rank(retrieved_chunks_list)[:self.top_k]

    def weighted_reciprocal_rank(self, retrieved_chunks_list: list[list[Chunk]]) -> list[Chunk]:
        if self.weights and len(self.weights) != len(retrieved_chunks_list):
            msg.fail("Weights length does not match the number of retrievers.")
            raise ValueError("Weights length does not match the number of retrievers")
        
        if self.weights is None:
            msg.warn("Weights are not provided. Using equal weights.")
            self.weights = [1.0] * len(retrieved_chunks_list)
        
        rrf_score: dict[str, float] = defaultdict(float)
        for chunks_list, weight in zip(retrieved_chunks_list, self.weights):
            for rank, chunk in enumerate(chunks_list, start=1):
                rrf_score[chunk.chunk_id] += weight / (rank + self.c)
        
        all_retrieved_chunks = chain.from_iterable(retrieved_chunks_list)
        sorted_chunks = sorted(
            unique_by_key(all_retrieved_chunks, key=lambda x: x.chunk_id),
            reverse=True,
            key=lambda x: rrf_score.get(x.chunk_id, 0)
        )
        return sorted_chunks