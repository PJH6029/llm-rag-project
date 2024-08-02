from wasabi import msg
import os

from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.documents import Document

from rag.component.retriever.base import BaseRAGRetriever
from rag.type import *
from rag import util

class RetrievalConfig:
    def __init__(self, config: dict):
        self._config = config
    
    # implementation of AmazonKnowledgeBasesRetriever's retrieval_config property
    def dict(self): 
        return self._config

class KnowledgeBaseRetriever(BaseRAGRetriever):
    def __init__(self, top_k: int = 5, **kwargs) -> None:
        super().__init__(top_k)
        
        required = ["knowledge_base_id", "region_name"]
        for attr in required:
            assert hasattr(self, attr), f"KnowledgeBaseRetriever subclass must have {attr} attribute"
            
        self.retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=self.knowledge_base_id,
            region_name=self.region_name,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": self.top_k,
                }
            }
        )
    
    def _set_env(self):
        """Override this method to set knowledge_base_id and region_name.
        """
        self.knowledge_base_id = os.environ["KNOWLEDGE_BASE_ID"]
        self.region_name = os.environ["AWS_REGION"]
        
        
    
    def retrieve(self, queries: TransformationResult, filter: Filter | None = None) -> list[Chunk]:
        _queries = util.flatten_queries(queries)
        
        if filter is not None:
            filter_dict = self._arange_filter(filter)
            self.retriever.retrieval_config = RetrievalConfig({
                "vectorSearchConfiguration": {
                    "numberOfResults": self.top_k,
                    "filter": filter_dict
                }
            })
        else:
            self.retriever.retrieval_config = RetrievalConfig({
                "vectorSearchConfiguration": {
                    "numberOfResults": self.top_k,
                }
            })
        retrieved_chunks_raw = self.retriever.batch(_queries)
        retrieved_chunks_raw = sum(retrieved_chunks_raw, [])
        retrieved_chunks = [self.process_chunk(chunks_raw) for chunks_raw in retrieved_chunks_raw]
        retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x.score, reverse=True)[:self.top_k]
        return retrieved_chunks

    def _arange_filter(self, filter: Filter) -> dict:
        key_map = {
            "base_doc_id": "base-doc-id",
        }
        
        if isinstance(filter, FilterPredicate):
            key = key_map.get(filter.key, filter.key)
            return {
                filter.op: {
                    "key": key,
                    "value": filter.value
                }
            }
        elif isinstance(filter, FilterExpression):
            return {
                filter.op: [
                    self._arange_filter(predicate) for predicate in filter.predicates
                ]
            }
        else:
            # should never reach here
            raise ValueError("Invalid filter")
    
    def _get_chunk_id(self, chunk_raw: Document) -> str:
        if chunk_raw.metadata["source_metadata"].get("x-amz-bedrock-kb-chunk-id"):
            return chunk_raw.metadata["source_metadata"]["x-amz-bedrock-kb-chunk-id"]
        else:
            return util.generate_id(chunk_raw.page_content)
    
    def process_chunk(self, chunk_raw: Document) -> Chunk:
        doc_meta, chunk_meta = self._process_metadata(chunk_raw)
        return Chunk(
            text=chunk_raw.page_content,
            doc_id=chunk_raw.metadata["source_metadata"]["x-amz-bedrock-kb-source-uri"],
            chunk_id=self._get_chunk_id(chunk_raw),
            doc_meta=doc_meta,
            chunk_meta=chunk_meta,
            score=chunk_raw.metadata["score"],
            source_retriever=self.__class__.__name__
        )
    
    def _process_metadata(self, chunk_raw: Document) -> tuple[dict, dict]:
        metadata = chunk_raw.metadata
        doc_meta = {
            "doc_id": metadata["source_metadata"]["x-amz-bedrock-kb-source-uri"],
            "doc_name": metadata["source_metadata"]["x-amz-bedrock-kb-source-uri"].split("/")[-1],
            "category": metadata["source_metadata"]["category"],
            "version": metadata["source_metadata"]["version"],
            "uri": metadata["source_metadata"]["x-amz-bedrock-kb-source-uri"],
        }
        if metadata["source_metadata"].get("base-doc-id"):
            doc_meta["base_doc_id"] = metadata["source_metadata"]["base-doc-id"]
        
        chunk_meta = {
            "chunk_id": self._get_chunk_id(chunk_raw),
            "score": metadata["score"],
        }
        return doc_meta, chunk_meta

    @classmethod
    def from_config(cls, config: dict) -> BaseRAGRetriever:
        top_k = config.get("top_k", cls.DEFAULT_TOP_K)
        return cls(top_k=top_k)

class KnowledgeBaseOpenSearchRetriever(KnowledgeBaseRetriever):
    def _set_env(self):
        self.knowledge_base_id = os.environ["KNOWLEDGE_BASE_OSS_ID"]
        self.region_name = os.environ["AWS_REGION"]

class KnowledgeBasePineconeRetriever(KnowledgeBaseRetriever):
    def _set_env(self):
        self.knowledge_base_id = os.environ["KNOWLEDGE_BASE_PINECONE_ID"]
        self.region_name = os.environ["AWS_REGION"]
