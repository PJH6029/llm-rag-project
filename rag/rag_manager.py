from typing import Generator, Optional
from wasabi import msg
import time
import os

from langchain_core.documents import Document

from rag.managers import (
    BasePipelineManager,
    TransformerManager,
    RetrieverManager,
    GeneratorManager,
    FactVerifierManager,
    IngestorManager
)
from rag.type import *
from rag import util
from rag.component import chunker, loader

class RAGManager:
    def __init__(self) -> None:
        self.ingestor_manager = IngestorManager()
        self.transformer_manager = TransformerManager()
        self.retriever_manager = RetrieverManager()
        self.generator_manager = GeneratorManager()
        self.fact_verifier_manager = FactVerifierManager()
        
        self.config = {}
    
    def set_config(self, config: dict):
        self.config = {**config}
        self.global_config = config.get("global", {})
        
        config_sections = ["ingestion", "transformation", "retrieval", "generation", "fact_verification"]
        managers: list[BasePipelineManager] = [self.ingestor_manager, self.transformer_manager, self.retriever_manager, self.generator_manager, self.fact_verifier_manager]
        for section, manager in zip(config_sections, managers):
            manager.set_config(util.merge_configs(config.get(section, {}), self.global_config))
        msg.good("RAGManager successfully configured")
        
    def transform_query(self, query: str, history: list[ChatLog]) -> TransformationResult:
        msg.info(f"Transforming query starting with: '{query}' and {len(history)} history...")
        start = time.time()
        
        queries = self.transformer_manager.transform(query, history)
        
        end = time.time()
        msg.good(f"Query transformed in {end-start:.2f}s, resulting in {len(queries)} queries")
        return queries

    def retrieve(self, queries: TransformationResult) -> list[Chunk]:
        msg.info(f"Retrieving with: {len(queries)} queries...")
        msg.info(f"Queries: {queries}")
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
        
        context = util.format_chunks(chunks or [], self.global_config.get("context-hierarchy", False))
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
        
        context = util.format_chunks(chunks or [], self.global_config.get("context-hierarchy", False))
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
        
        context = util.format_chunks(chunks or [], self.global_config.get("context-hierarchy", False))
        verification_response = self.fact_verifier_manager.verify(response, context)
        
        end = time.time()
        msg.good(f"Fact verification completed in {end-start:.2f}s")
        return verification_response
    
    def verify_fact_stream(self, response: str, chunks: list[Chunk]) -> Generator[str, None, None]:
        msg.info(f"Verifying fact...")
        start = time.time()
        
        context = util.format_chunks(chunks or [], self.global_config.get("context-hierarchy", False))
        for r in self.fact_verifier_manager.verify_stream(response, context):
            yield r
        
        end = time.time()
        msg.good(f"Fact verification completed in {end-start:.2f}s")
        
    def ingest(self, file_path: str, batch_size: int = 20) -> int:
        msg.info(f"Ingesting data from {file_path}")
        start = time.time()
                
        chunks_iter = loader.lazy_load(file_path)
        
        msg.good(f"Loaded documents")

        chunks_cnt = util.execute_as_batch(
            chunks_iter,
            batch_size=batch_size,
            func=self.ingestor_manager.ingest
        )
        
        end = time.time()
        msg.good(f"{chunks_cnt} chunks ingested in {end-start:.2f}s")
        return chunks_cnt
    
    def aingest(self, data_url: str, batch_size: int = 20) -> int:
        # TODO
        return -1
    
    def ingest_from_backup(
        self, backup_dir: str, object_location: str, batch_size: int = 20
    ) -> int:
        msg.info(f"Ingesting data from {backup_dir}")
        start = time.time()
        
        chunks_iter = loader.lazy_load_from_backup(backup_dir, object_location)
        
        chunks_cnt = util.execute_as_batch(
            chunks_iter,
            batch_size=batch_size,
            func=self.ingestor_manager.ingest
        )
        
        end = time.time()
        msg.good(f"{chunks_cnt} chunks ingested in {end-start:.2f}s")
        return chunks_cnt

    def upload_data(self, file_path: str, object_location: str, metadata: Optional[dict] = None) -> bool:
        msg.info(f"Uploading data from {file_path} to {object_location}")
        start = time.time()
        
        success = util.upload_to_s3_with_metadata(file_path, object_location=object_location, metadata=metadata)
        
        end = time.time()
        msg.good(f"Data uploaded in {end-start:.2f}s")
        return success