from wasabi import msg
import os

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings

from rag.model.retrievers.base import BaseRAGRetriever
from rag.type import *
from rag.util import generate_id

class PineconeRetriever(BaseRAGRetriever):
    def __init__(self, top_k: int = 5, embeddings: Embeddings = None) -> None:
        super().__init__(top_k, embeddings)
        self.vectorstore = PineconeVectorStore(
            index_name = self.index_name,
            embedding=self.embeddings,
        )
    
    def _set_env(self):
        self.index_name = os.environ["PINECONE_INDEX_NAME"]

    def retrieve(self, queries: list[str], filter: Filter | None = None) -> list[Chunk]:
        if filter is not None:
            filter_dict = self._arange_filter(filter)
        
        retrieved_chunks = []
        for query in queries:
            result = self.vectorstore.similarity_search_with_score(
                query=query,
                k=self.top_k,
                filter=filter_dict,
            )
            if result:
                retrieved_chunks_raw, scores = list(zip(*result))
            else:
                retrieved_chunks_raw, scores = [], []
            for idx, chunks_raw in enumerate(retrieved_chunks_raw):
                chunks_raw.metadata["score"] = scores[idx]
            retrieved_chunks_for_query = [self.process_chunk(chunks_raw) for chunks_raw in retrieved_chunks_raw]
            retrieved_chunks.extend(retrieved_chunks_for_query)
        return retrieved_chunks

    def _arange_filter(self, filter: Filter) -> dict:
        op_map = {
            "equals": "$eq",
            "notEquals": "$ne",
            "greaterThan": "$gt",
            "greaterThanOrEquals": "$gte",
            "lessThan": "$lt",
            "lessThanOrEquals": "$lte",
            "in": "$in",
            "notIn": "$nin",
            "andAll": "$and",
            "orAll": "$or",
        }
        
        key_map = {
            "base_doc_id": "base-doc-id",
        }
        
        if isinstance(filter, FilterPredicate):
            key = key_map.get(filter.key, filter.key)
            return {
                key: {
                    op_map[filter.op]: filter.value
                }
            }
        elif isinstance(filter, FilterExpression):
            return {
                op_map[filter.op]: [self._arange_filter(pred) for pred in filter.predicates]
            }
    
    def process_chunk(self, chunk_raw: Document) -> Chunk:
        doc_meta, chunk_meta = self._process_metadata(chunk_raw)
        return Chunk(
            text=chunk_raw.page_content,
            doc_id=chunk_raw.metadata["x-amz-bedrock-kb-source-uri"],
            chunk_id=generate_id(chunk_raw.page_content),
            doc_meta=doc_meta,
            chunk_meta=chunk_meta,
            score=chunk_raw.metadata["score"]
        )
    
    def _process_metadata(self, chunk_raw: Document) -> tuple[dict, dict]:
        metadata = chunk_raw.metadata
        doc_meta = {
            "doc_id": metadata["x-amz-bedrock-kb-source-uri"],
            "doc_name": metadata["x-amz-bedrock-kb-source-uri"].split("/")[-1],
            "category": metadata["category"],
            "version": metadata["version"],
            "uri": metadata["x-amz-bedrock-kb-source-uri"],
        }
        if metadata.get("base-doc-id"):
            doc_meta["base_doc_id"] = metadata["base-doc-id"]
        
        chunk_meta = {
            "chunk_id": generate_id(chunk_raw.page_content),
            "score": metadata["score"],
        }
        return doc_meta, chunk_meta