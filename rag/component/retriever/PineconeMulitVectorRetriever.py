import os
from typing import Optional
from langchain_core.documents import Document
from wasabi import msg

from pinecone import Pinecone, Index

from rag.component.vectorstore.base import BaseRAGVectorstore
from rag.component.vectorstore.PineconeVectorstore import PineconeVectorstore
from rag.component.retriever.base import BaseRAGRetriever
from rag.component import embeddings
from rag.type import *
from rag.type import Chunk, Document
from rag import util

class PineconeMultiVectorRetriever(BaseRAGRetriever):
    PARENT_CHILD_FACTOR = 3
    def __init__(
        self, 
        vectorstore: PineconeVectorstore,
        sub_vectorstore: PineconeVectorstore,
        top_k: int = 5,
        parent_id_key: str = "parent_id",
        **kwargs
    ) -> None:
        super().__init__(top_k, **kwargs)
        
        self.vectorstore = vectorstore
        self.sub_vectorstore = sub_vectorstore
        self._parent_id_key = parent_id_key
    
    def retrieve(self, queries: list[str], filter: Filter | None = None) -> list[Chunk]:  
        try:
            if filter is not None:
                filter_dict = self._arange_filter(filter)
            else:
                filter_dict = None

            id_scores = dict()
            sub_chunk_cnt = 0
            for query in queries:
                sub_chunks = self.sub_vectorstore.query(query, top_k=self.top_k * self.PARENT_CHILD_FACTOR, filter=filter_dict)
                sub_chunk_cnt += len(sub_chunks)
                
                for sub_chunk in sub_chunks:
                    if self._parent_id_key in sub_chunk.chunk_meta:
                        if sub_chunk.chunk_meta[self._parent_id_key] not in id_scores:
                            id_scores[sub_chunk.chunk_meta[self._parent_id_key]] = []
                        id_scores[sub_chunk.chunk_meta[self._parent_id_key]].append(sub_chunk.score)
            
            # retrieve parent chunks
            retrieved_chunks_raw = self.vectorstore.fetch_docs(list(id_scores.keys()))
            if not retrieved_chunks_raw:
                msg.warn(f"No parent chunks retrieved, based on {sub_chunk_cnt} sub chunks")
                return []
            
            # normalize scores using min-max scaling
            for key in id_scores:
                id_scores[key] = sum(id_scores[key])
            min_score = min(id_scores.values())
            max_score = max(id_scores.values())
            for key in id_scores:
                id_scores[key] = (id_scores[key] - min_score) / (max_score - min_score)
            
            # assign scores
            for retrieved_chunk_raw in retrieved_chunks_raw:
                chunk_id = retrieved_chunk_raw.metadata["chunk_id"]
                if chunk_id in id_scores:
                    retrieved_chunk_raw.metadata["score"] = id_scores[chunk_id]
                else:
                    # This should not happen
                    msg.warn(f"Chunk {chunk_id} is retrieved even though it has no retrieved sub chunks")
                    retrieved_chunk_raw.metadata["score"] = 0 
            
            retrieved_chunks = [self.process_chunk(chunk_raw) for chunk_raw in retrieved_chunks_raw]
            retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x.score, reverse=True)[:self.top_k]
            msg.info(f"Retrieved {len(retrieved_chunks)} chunks from parent vectorstore, based on {sub_chunk_cnt} sub chunks")
            return retrieved_chunks
        except Exception as e:
            msg.warn(f"Error occurred during retrieval using {self.__class__.__name__}: {e}")
        

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
        
        # TODO seperator
        key_map = {
            "base_doc_id": "doc_meta___Attributes___base-doc-id",
            "category": "doc_meta___Attributes____category",
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
        metadata = util.deflatten_dict(chunk_raw.metadata)
        doc_meta = metadata.get("doc_meta", {})
        
        chunk_meta = metadata.get("chunk_meta", {})
        doc_name = doc_meta.get("doc_id", "").split("/")[-1]
        return Chunk(
            text=chunk_raw.page_content,
            chunk_id=metadata.get("chunk_id"),
            doc_id=metadata.get("doc_id"),
            doc_meta=util.remove_falsy({
                "doc_name": doc_name,
                "category": doc_meta.get("Attributes", {}).get("_category"),
                "base_doc_id": doc_meta.get("Attributes", {}).get("base-doc-id"),
                "version": doc_meta.get("Attributes", {}).get("version"),
            }),
            chunk_meta={**chunk_meta, "score": metadata.get("score")},
            score=metadata.get("score"),
        )
        
    @classmethod
    def from_config(cls, config: dict) -> "PineconeMultiVectorRetriever":
        top_k = config.get("top_k", cls.DEFAULT_TOP_K)
        parent_namespace = config.get("namespace")
        child_namespace = config.get("sub-namespace")
        
        embeddings_name = config.get("embeddings")
        embedding_model = embeddings.get_model(embeddings_name)
        
        vectorstore = PineconeVectorstore(
            embeddings=embedding_model,
            namespace=parent_namespace,
        )
        
        sub_vectorstore = PineconeVectorstore(
            embeddings=embedding_model,
            namespace=child_namespace,
        )
        
        return cls(
            vectorstore=vectorstore,
            sub_vectorstore=sub_vectorstore,
            top_k=top_k,
        )