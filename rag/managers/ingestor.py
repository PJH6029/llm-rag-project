from wasabi import msg
from typing import Optional, Type, Iterable

from langchain_core.embeddings import Embeddings

from rag.type import *
from rag.managers.base import BasePipelineManager
from rag.component import embeddings
from rag.component.ingestor import *
from rag import util

class IngestorManager(BasePipelineManager):
    def __init__(self) -> None:
        super().__init__()
        # self.embeddings_name: Optional[str] = None
        
        self.ingestors: dict[str, Type[BaseRAGIngestor]] = {
            "pinecone": PineconeVectorstoreIngestor,
            "pinecone-multivector": PineconeMultiVectorIngestor,
        }
        self.selected_ingestors: Optional[BaseRAGIngestor] = None
        
        
    def set_config(self, config: dict):
        self.ingestor_name = config.get("ingestor")
        # self.embeddings_name = config.get("embedding")
        
        if self.ingestor_name is None:
            msg.warn("INGESTOR not configured. Setting to None.")
        else:
            msg.info(f"Setting INGESTOR to {self.ingestor_name}")
            
        self.init_ingestor(config)
        
    def init_ingestor(self, config: dict):
        if self.ingestor_name is None:
            return
        
        self.selected_ingestors = self.ingestors[self.ingestor_name].from_config(config)
            
    def ingest(self, chunks: list[Chunk]) -> bool:
        ingestor = self.selected_ingestors
        if ingestor is None:
            msg.warn("No ingestor initialized. Skipping ingestion.")
            return False
        
        return ingestor.ingest(chunks)
