from typing import Generator, Optional
from wasabi import msg
import time

from rag.managers import (
    BasePipelineManager,
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
        self.global_config = config.get("global", {})
        
        config_sections = ["transformation", "retrieval", "generation", "fact_verification"]
        managers: list[BasePipelineManager] = [self.transformer_manager, self.retriever_manager, self.generator_manager, self.fact_verifier_manager]
        for section, manager in zip(config_sections, managers):
            manager.set_config(util.merge_configs(config.get(section, {}), self.global_config))
        msg.good("RAGManager successfully configured")
        
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
        
        chunks = self.retriever_manager.retrieve(queries)
        
        end = time.time()
        msg.good(f"{len(chunks)} chunks retrieved in {end-start:.2f}s")
        return chunks

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