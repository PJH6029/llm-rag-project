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
    IngestorManager
)
from rag.type import *
from rag import util
from rag.component import chunker, loader

class time_logger:
    def __init__(self, start_msg_callback: Callable, end_msg_callback: Callable) -> None:
        self.start_msg_callback = start_msg_callback
        self.end_msg_callback = end_msg_callback
        self.start = 0
    
    def __enter__(self) -> None:
        msg.info(self.start_msg_callback())
        self.start = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        end = time.time()
        msg.good(f"{self.end_msg_callback()} ({end - self.start:.2f}s)")
        
class Managers(TypedDict):
    ingestor: IngestorManager
    transformer: TransformerManager
    retriever: RetrieverManager
    generator: GeneratorManager
    fact_verifier: FactVerifierManager
    
ManagerKey = Literal["ingestor", "transformer", "retriever", "generator", "fact_verifier"]
ManagerValue = Union[IngestorManager, TransformerManager, RetrieverManager, GeneratorManager, FactVerifierManager]

class ManagersDict():
    @classmethod
    def from_dict(cls, dict: Managers) -> "ManagersDict":
        managers = ManagersDict()
        for k, v in dict.items():
            managers[k] = v
        return managers
    
    def __init__(self) -> None:
        self.ingestion: Optional[IngestorManager] = None
        self.transformation: Optional[TransformerManager] = None
        self.retrieval: Optional[RetrieverManager] = None
        self.generation: Optional[GeneratorManager] = None
        self.fact_verification: Optional[FactVerifierManager] = None
    
    def __getitem__(self, key: ManagerKey) -> ManagerValue:
        return getattr(self, key)

    def __setitem__(self, key: ManagerKey, value: ManagerValue) -> None:
        setattr(self, key, value)
        
    def items(self) -> list[tuple[ManagerKey, ManagerValue]]:
        return [(k, v) for k, v in self.__dict__.items() if isinstance(v, ManagerValue)]

class RAGManager:
    def __init__(self) -> None:
        self.managers = ManagersDict.from_dict({
            "ingestion": IngestorManager(),
            "transformation": TransformerManager(),
            "retrieval": RetrieverManager(),
            "generation": GeneratorManager(),
            "fact_verification": FactVerifierManager()
        })
        
        self.config = {}
    
    def set_config(self, config: dict):
        self.config = {**config}
        self.global_config = config.get("global", {})
        
        for manager_key, manager in self.managers.items():
            manager.set_config(util.merge_configs(config.get(manager_key, {}), self.global_config))
        msg.good("RAGManager successfully configured")
        
    def transform_query(self, query: str, history: list[ChatLog]) -> TransformationResult:
        with time_logger(
            lambda: f"Transforming query `{query}` with {len(history)} history...",
            lambda: f"Query transformed into {len(queries)} queries"
        ):
            queries = self.managers.transformation.transform(query, history)
            msg.info(f"Transformed queries: {queries}")
            return queries

    def retrieve(self, queries: TransformationResult) -> list[Chunk]:
        with time_logger(
            lambda: f"Retrieving with {len(queries)} queries...",
            lambda: f"{len(chunks)} chunks retrieved"
        ):
            chunks = self.managers.retrieval.retrieve(queries)
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
            context = util.format_chunks(chunks, self.global_config.get("context-hierarchy", False))
            history_str = util.format_history(history)
            
            generation_response = self.managers.generation.generate(query, history_str, context)
            
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
            context = util.format_chunks(chunks, self.global_config.get("context-hierarchy", False))
            history_str = util.format_history(history)
            
            yield from self.managers.generation.generate_stream(query, history_str, context)
    
    def verify_fact(self, response: str, chunks: list[Chunk]) -> VerificationResult:
        with time_logger(
            lambda: f"Verifying fact...",
            lambda: f"Fact verification completed"
        ):
            context = util.format_chunks(chunks or [], self.global_config.get("context-hierarchy", False))
            verification_response = self.managers.fact_verification.verify(response, context)
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
                func=self.managers.ingestion.ingest
            )
            return chunks_cnt
    
    def ingest(self, file_path: str, batch_size: int = 20) -> int:
        chunks_iter = loader.lazy_load(file_path)
        return self._ingest_with_loader(chunks_iter, batch_size=batch_size)
    
    def aingest(self, data_url: str, batch_size: int = 20) -> int:
        # TODO
        return -1
    
    def ingest_from_backup(
        self, backup_dir: str, object_location: str, batch_size: int = 20
    ) -> int:
        chunks_iter = loader.lazy_load_from_backup(backup_dir, object_location)
        return self._ingest_with_loader(chunks_iter, batch_size=batch_size)

    def upload_data(self, file_path: str, object_location: str) -> bool:
        with time_logger(
            lambda: f"Uploading data from {file_path} to {object_location}...",
            lambda: f"Data uploaded"
        ):
            success = util.upload_to_s3_with_metadata(file_path, object_location=object_location)
            return success
