import os

from langchain_community.retrievers import AmazonKendraRetriever
from langchain_core.documents import Document

from rag.model.retrievers.base import BaseRAGRetriever
from rag.type import *

class KendraRetriever(BaseRAGRetriever):
    def __init__(self, top_k: int = 5, embeddings = None) -> None:
        super().__init__(top_k)
        self.retriever = AmazonKendraRetriever(
            index_id = self.kendra_index_id,
            region_name = self.region_name,
            top_k = self.top_k,
        )
    
    def _set_env(self):
        self.kendra_index_id = os.environ["KENDRA_INDEX_ID"]
        self.region_name = os.environ["AWS_REGION"]
    
    def retrieve(self, queries: list[str], filter: Filter | None = None) -> list[Chunk]:
        if filter is not None:
            filter_dict = self._arange_filter(filter)
            self.retriever.attribute_filter = filter_dict

        retrieved_chunks_raw = self.retriever.batch(queries)
        retrieved_chunks_raw = sum(retrieved_chunks_raw, [])
        retrieved_chunks = [self.process_chunk(chunks_raw) for chunks_raw in retrieved_chunks_raw]
        return retrieved_chunks
    
    def _arange_filter(self, filter: Filter) -> dict:
        op_map = {
            "equals": "EqualsTo",
            "notEquals": "EqualsTo", # will be wrapped by NotFilter
            "greaterThan": "GreaterThan",
            "greaterThanOrEquals": "GreaterThanOrEquals",
            "lessThan": "LessThan",
            "lessThanOrEquals": "LessThanOrEquals",
            "andAll": "AndAllFilters",
            "orAll": "OrAllFilters",
        }
        
        key_map = {
            "category": "_category",
            "base_doc_id": "base-doc-id",
        }
        
        if isinstance(filter, FilterPredicate):
            key = key_map.get(filter.key, filter.key)
            if filter.op in ["equals", "greaterThan", "greaterThanOrEquals", "lessThan", "lessThanOrEquals"]:
                value_type = "StringValue" if isinstance(filter.value, str) else "StringListValue" if isinstance(filter.value, list) else "Long"
                return {op_map[filter.op]: {"Key": key, "Value": {value_type: filter.value}}}
            elif filter.op in ["notEquals"]:
                value_type = "StringValue" if isinstance(filter.value, str) else "StringListValue" if isinstance(filter.value, list) else "Long"
                return {"NotFilter": {op_map[filter.op]: {"Key": key, "Value": {value_type: filter.value}}}}                
            elif filter.op in ["in"]:
                return {"OrAllFilters": [{"EqualsTo": {"Key": key, "Value": {"StringValue": item}}} for item in filter.value]}
            elif filter.op in ["notIn"]:
                return {"NotFilter": {"OrAllFilters": [{"EqualsTo": {"Key": key, "Value": {"StringValue": item}}} for item in filter.value]}}
            elif filter.op in ["startsWith"]:
                raise ValueError("Operation 'startsWith' is not supported in Kendra")
        elif isinstance(filter, FilterExpression):
            return {
                op_map[filter.op]: [
                    self._arange_filter(predicate) for predicate in filter.predicates
                ]
            }
    
    def _score_floatify(self, score: str) -> float:
        if score == "VERY_HIGH":
            return 1.0
        elif score == "HIGH":
            return 0.8
        elif score == "MEDIUM":
            return 0.5
        elif score == "LOW":
            return 0.3
        else:
            return 0.1
        
    def process_chunk(self, chunk_raw: Document) -> Chunk:
        doc_meta, chunk_meta = self._process_metadata(chunk_raw)
        return Chunk(
            text=chunk_raw.metadata["excerpt"],
            doc_id=chunk_raw.metadata["document_id"],
            chunk_id=chunk_raw.metadata["result_id"],
            doc_meta=doc_meta,
            chunk_meta=chunk_meta,
            score=self._score_floatify(chunk_raw.metadata["score"]),
        )

    def _process_metadata(self, chunk_raw: Document) -> tuple[dict, dict]:
        metadata = chunk_raw.metadata
        doc_meta = {
            "doc_id": metadata.get("document_id", ""),
            "doc_name": metadata.get("title", ""),
            "category": metadata.get("document_attributes", {}).get("_category", ""),
            "version": metadata.get("document_attributes", {}).get("version", ""),
            "uri": metadata.get("document_attributes", {}).get("_source_uri", ""),
        }
        if metadata.get("document_attributes", {}).get("base-doc-id"):
            doc_meta["base_doc_id"] = metadata.get("document_attributes", {}).get("base-doc-id")
        
        chunk_meta = {
            "chunk_id": metadata.get("result_id", ""),
            "score": metadata.get("score", ""),
            "page": metadata.get("document_attributes", {}).get("_excerpt_page_number", -1),
        }
        return doc_meta, chunk_meta