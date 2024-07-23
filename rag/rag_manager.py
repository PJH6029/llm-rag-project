from typing import Generator, Optional
from wasabi import msg
import time

from rag.managers import (
    TransformerManager,
    RetrieverManager,
    GeneratorManager,
    FactVerifierManager,
)
from rag.type import *
from rag import util

class RAGManager:
    def __init__(self) -> None:
        self.transformer_manager = TransformerManager()
        self.retriever_manager = RetrieverManager()
        self.generator_manager = GeneratorManager()
        self.fact_verifier_manager = FactVerifierManager()
        
        self.config = {}
    
    def set_config(self, config: dict):
        self.config = {**config}
        self.transformer_manager.set_config(config.get("transformation", {}))
        self.retriever_manager.set_config(config.get("retrieval", {}))
        self.generator_manager.set_config(config.get("generation", {}))
        self.fact_verifier_manager.set_config(config.get("fact_verification", {}))
    
    def transform_query(self, query: str, history: list[ChatLog]) -> list[str]:
        msg.info(f"Transforming query starting with: '{query}' and {len(history)} history...")
        start = time.time()
        
        queries = self.transformer_manager.transform(query, history)
        
        end = time.time()
        msg.good(f"Query transformed in {end-start:.2f}s, resulting in {len(queries)} queries")
        return queries
    
    def retrieve(self, queries: list[str]) -> list[Chunk]:
        msg.info(f"Retrieving with: {len(queries)} queries...")
        start = time.time()
        
        # base context retrieval
        base_chunks = self.retriever_manager.retrieve(
            queries, filter={"equals": {"key": "category", "value": "base"}}
        )
        managed_base_chunks = self.retriever_manager.rerank(base_chunks) # TODO topk
        
        # additional context retrieval
        base_doc_ids = list(set([c.doc_id for c in managed_base_chunks]))
        additional_chunks = self.retriever_manager.retrieve(
            queries, filter={
                "andAll": [
                    {"equals": {"key": "category", "value": "additional"}}, 
                    {"in": {"key": "base_doc_id", "value": base_doc_ids}}
                ]
            }
        )
        managed_additional_chunks = self.retriever_manager.rerank(additional_chunks) # TODO topk
        
        invocation_cnt = 2 * len(queries)
        
        chunks = managed_base_chunks + managed_additional_chunks
        self.validate(chunks)
        
        end = time.time()
        msg.good(f"{len(chunks)} chunks retrieved in {end-start:.2f}s with {invocation_cnt} invocations")
        return chunks

    def validate(self, chunks: list[Chunk]) -> None:
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

    def generate(
        self, query: str, history: Optional[list[ChatLog]]=None, chunks: Optional[list[Chunk]]=None
    ) -> str:
        msg.info(f"Querying with: '{query}' and {len(history)} history...")
        start = time.time()
        
        context = util.format_chunks(chunks or [])
        history_str = util.format_history(history or [])
        
        generation_response = self.generator_manager.generate(query, history_str, context)
        
        end = time.time()
        msg.good(f"Query completed in {end-start:.2f}s")
        return generation_response
    
    def generate_stream(
        self, query: str, history: Optional[list[ChatLog]]=None, chunks: Optional[list[Chunk]]=None
    ) -> Generator[str, None, None]:
        msg.info(f"Querying with: {query} and {len(history)} history...")
        start = time.time()
        
        context = util.format_chunks(chunks or [])
        history_str = util.format_history(history or [])
                
        generation_response = ""
        for r in self.generator_manager.generate_stream(query, history_str, context):
            yield r
            generation_response += r
        
        end = time.time()
        msg.good(f"Query completed in {end-start:.2f}s")
    
    def verify_fact(self, response: str, chunks: list[Chunk]) -> str:
        msg.info(f"Verifying fact...")
        start = time.time()
        
        verification_response = self.fact_verifier_manager.verify(response, chunks)
        
        end = time.time()
        msg.good(f"Fact verification completed in {end-start:.2f}s")
        return verification_response
    
    def verify_fact_stream(self, response: str, chunks: list[Chunk]) -> Generator[str, None, None]:
        msg.info(f"Verifying fact...")
        start = time.time()
        
        for r in self.fact_verifier_manager.verify_stream(response, chunks):
            yield r
        
        end = time.time()
        msg.good(f"Fact verification completed in {end-start:.2f}s")