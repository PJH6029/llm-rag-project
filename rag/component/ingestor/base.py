from typing import Iterable

from rag.type import *

class BaseRAGIngestor:
    def __init__(self) -> None:
        pass
    
    def ingest(self, chunks: list[Chunk]) -> int:
        raise NotImplementedError()
    
    @classmethod
    def from_config(cls, config: dict) -> "BaseRAGIngestor":
        raise NotImplementedError()