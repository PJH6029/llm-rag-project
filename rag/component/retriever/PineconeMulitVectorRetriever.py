import os
from typing import Optional
from wasabi import msg

from pinecone import Pinecone, Index

from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, Runnable, RunnableLambda


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
    
    def _get_retrieval_chain(self, chain_num: int) -> Runnable:
        """Create a retrieval chain that retrieves chunks in parallel.

        Args:
            chain_num (int): The number of chains to create. Equals to the number of queries.

        Returns:
            RunnableParallel: The retrieval chain.
        """
        def _runnable_input_parser(_dict):
            queries = _dict["queries"]
            filter_dict = _dict["filter"]
            top_k = _dict["top_k"]

            result = {}
            for i in range(chain_num):
                result[f"query_{i}"] = queries[i]
            result["filter"] = filter_dict
            result["top_k"] = top_k
            return result
        runnable_input_parser = RunnableLambda(_runnable_input_parser)
        
        def _runnable_output_parser(_dict):
            result = sum([_dict[f"query_{i}"] for i in range(chain_num)], [])
            return result
        runnable_output_parser = RunnableLambda(_runnable_output_parser)
        
        def _build_query_chain(i: int) -> Runnable:            
            def _query_chain(_dict):
                query = _dict[f"query_{i}"]
                filter_dict = _dict["filter"]
                top_k = _dict["top_k"]
                
                return self.sub_vectorstore.query(query, top_k=top_k, filter=filter_dict)
            return RunnableLambda(_query_chain)
        
        parrallel_kwargs = {}
        for i in range(chain_num):
            parrallel_kwargs[f"query_{i}"] = _build_query_chain(i)
        
        runnable_parallel = RunnableParallel(**parrallel_kwargs)
        
        chain = runnable_input_parser | runnable_parallel | runnable_output_parser
        return chain
    
    def retrieve(self, queries: TransformationResult, filter: Filter | None = None) -> list[Chunk]:  
        try:
            _queries = util.flatten_queries(queries)
            
            if filter is not None:
                filter_dict = self._arange_filter(filter)
            else:
                filter_dict = None

            id_scores = dict()
            sub_chunk_cnt = 0
            
            sub_chunks = self._get_retrieval_chain(len(_queries)).invoke({"queries": _queries, "filter": filter_dict, "top_k": int(self.top_k * self.PARENT_CHILD_FACTOR)})
            sub_chunk_cnt = len(sub_chunks)
            
            for sub_chunk in sub_chunks:
                if self._parent_id_key in sub_chunk.chunk_meta:
                    if sub_chunk.chunk_meta[self._parent_id_key] not in id_scores:
                        id_scores[sub_chunk.chunk_meta[self._parent_id_key]] = []
                    id_scores[sub_chunk.chunk_meta[self._parent_id_key]].append(sub_chunk.score)
            
            # retrieve parent chunks
            retrieved_chunks_raw = self.vectorstore.fetch_docs(list(id_scores.keys()))
            if not retrieved_chunks_raw:
                msg.warn(f"Retrieved 0 chunks from parent vectorstore, based on {sub_chunk_cnt} sub chunks")
                return []
            
            # normalize scores using min-max scaling
            # TODO better normalization?
            for key in id_scores:
                id_scores[key] = sum(id_scores[key])
        
            if len(id_scores) == 1:
                # avoid division by zero
                for key in id_scores:
                    id_scores[key] = 1
            else: 
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
            return []
        

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
        doc_name = util.MetadataSearch.search_doc_id(metadata).split("/")[-1]
        return Chunk(
            text=chunk_raw.page_content,
            chunk_id=util.MetadataSearch.search_chunk_id(metadata),
            doc_id=util.MetadataSearch.search_doc_id(metadata),
            doc_meta=util.remove_falsy({
                "doc_id": util.MetadataSearch.search_doc_id(metadata),
                "doc_name": doc_name,
                "category": doc_meta.get("Attributes", {}).get("_category"),
                "base_doc_id": doc_meta.get("Attributes", {}).get("base-doc-id"),
                "version": doc_meta.get("Attributes", {}).get("version"),
            }),
            chunk_meta={**chunk_meta, "score": metadata.get("score"), "chunk_id": util.MetadataSearch.search_chunk_id(metadata)},
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