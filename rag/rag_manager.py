from typing import Generator, Iterable, Optional, Callable, Any, TypedDict
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
    IngestorManager,
    LoaderManager,
)
from rag.type import *
from rag import util
from rag.component import chunker, loader
from rag.util import time_logger
from rag.config import *
from rag.component.loader import BaseRAGLoader, BaseLoader
        
class Managers(TypedDict):
    load: LoaderManager
    ingestion: IngestorManager
    transformation: TransformerManager
    retrieval: RetrieverManager
    generation: GeneratorManager
    fact_verification: FactVerifierManager
    
    def __getitem__(self, key: str) -> BasePipelineManager:
        return self.get(key)

class RAGManager:
    def __init__(self) -> None:
        self.managers: Managers = {
            "load": LoaderManager(),
            "ingestion": IngestorManager(),
            "transformation": TransformerManager(),
            "retrieval": RetrieverManager(),
            "generation": GeneratorManager(),
            "fact_verification": FactVerifierManager(),
        }
        
        self.config: Optional[RAGConfig] = None
    
    def set_config(self, config: RAGConfig) -> None:
        print(config)
        self.config = config
        self.global_config = config.global_
        
        for manager_key, manager in self.managers.items():
            manager.set_config(util.attach_global_config(getattr(self.config, manager_key), self.global_config))
        msg.good("RAGManager successfully configured")
        
    def transform_query(self, query: str, history: list[ChatLog]) -> TransformationResult:
        with time_logger(
            lambda: f"Transforming query: {query} with {len(history)} history...",
            lambda: f"Query transformed into {len(queries)} queries"
        ):
            queries = self.managers["transformation"].transform(query, history)
            msg.info(f"Transformed queries: {queries}")
            return queries

    def retrieve(self, queries: TransformationResult, categories: list[str]=None) -> list[Chunk]:
        with time_logger(
            lambda: f"Retrieving with {len(queries)} queries...",
            lambda: f"{len(chunks)} chunks retrieved"
        ):
            chunks = self.managers["retrieval"].retrieve(
                queries,
                filter = {"in": {"key": "category", "value": categories}} if categories else None
            )
            return chunks

    def generate(
        self, 
        query: str, 
        history: Optional[list[ChatLog]] = None, 
        chunks: Optional[list[Chunk]] = None
    ) -> str:
        chunks = chunks or []
        history = history or []
        
        with time_logger(
            lambda: f"Querying with: `{query}` and {len(history)} history...",
            lambda: f"Query completed"
        ):
            context = util.format_chunks(chunks, self.global_config.context_hierarchy)
            history_str = util.format_history(history)
            
            generation_response = self.managers["generation"].generate(query, history_str, context)
            return generation_response
    
    def generate_stream(
        self, 
        query: str, 
        history: Optional[list[ChatLog]] = None, 
        chunks: Optional[list[Chunk]] = None
    ) -> Generator[str, None, None]:
        chunks = chunks or []
        history = history or []
        
        with time_logger(
            lambda: f"Querying with: `{query}` and {len(history)} history...",
            lambda: f"Query completed"
        ):
            context = util.format_chunks(chunks, self.global_config.context_hierarchy)
            history_str = util.format_history(history)
            
            yield from self.managers["generation"].generate_stream(query, history_str, context)
    
    def verify_fact(self, response: str, chunks: list[Chunk]) -> Optional[VerificationResult]:
        with time_logger(
            lambda: f"Verifying fact...",
            lambda: f"Fact verification completed"
        ):
            context = util.format_chunks(chunks or [], self.global_config.context_hierarchy)
            verification_response = self.managers["fact_verification"].verify(response, context)
            return verification_response

    
    # def verify_fact_stream(self, response: str, chunks: list[Chunk]) -> Generator[str, None, None]:
    #     with time_logger(
    #         lambda: f"Verifying fact...",
    #         lambda: f"Fact verification completed"
    #     ):
    #         context = util.format_chunks(chunks or [], self.global_config.get("context-hierarchy", False))
    #         yield from self.managers.fact_verification.verify_stream(response, context)
    
    def _ingest_with_loader(self, loader: Iterable[Chunk], batch_size: int = 20) -> int:
        with time_logger(
            lambda: f"Ingesting data...",
            lambda: f"Data ingested"
        ):
            chunks_cnt = util.execute_as_batch(
                loader,
                batch_size=batch_size,
                func=self.managers["ingestion"].ingest
            )
            return chunks_cnt
    
    # TODO route loader. For now, specific loader should be passed
    def ingest(
        self, 
        # resource_path: Any, 
        loader: Union[BaseRAGLoader, BaseLoader],
        batch_size: int = 20
    ) -> int:
        chunks_iter = self.managers["load"].lazy_load_chunk(loader=loader)
        return self._ingest_with_loader(chunks_iter, batch_size=batch_size)
    
    def aingest(self, data_url: str, batch_size: int = 20) -> int:
        # TODO
        return -1

    def upload_data(self, file_path: str, object_location: str, metadata: Optional[dict] = None) -> bool:
        """Uploads data to S3
        """
        with time_logger(
            lambda: f"Uploading data from {file_path} to {object_location}...",
            lambda: f"Data uploaded"
        ):
            success = util.upload_to_s3_with_metadata(file_path, object_location=object_location, metadata=metadata)
            return success


    def get_categories(self) -> list[str]:
        return [
            "APPLE",
            "Google",
            "MCTP",
            "MS",
            "NVMe",
            "NVMe-MI",
            "OCP/2.0",
            "OCP/2.5",
            "PCIe",
        ]