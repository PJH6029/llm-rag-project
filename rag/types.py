from pydantic import BaseModel
from typing import Literal

class Chunk:
    def __init__(
        self,
        text: str="",
        doc_id: str="",
        chunk_id: str="",
        doc_meta: dict=None,
        chunk_meta: dict=None,
    ):
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = chunk_id

        self.tokens = 0
        self.score = 0.0

        if doc_meta is None:
            doc_meta = {}
        self.doc_meta = doc_meta

        if chunk_meta is None:
            chunk_meta = {}
        self.chunk_meta = chunk_meta

    
    # def to_dict(self) -> dict:
    #     return {
    #         "text": self.text,
    #         "doc_name": self.doc_name,
    #         "doc_type": self.doc_type,
    #         "doc_id": self.doc_id,
    #         "chunk_id": self.chunk_id,
    #         "tokens": self.tokens,
    #         "score": self.score,
    #         "meta": self.meta,
    #     }
    
    def to_detailed_str(self):
        return (
            f"CHUNK ({self.chunk_id})\n"
            f"(\n"
            f"SIMILARITY_SCORE= {self.score}\n"
            f"TEXT=\n"
            f"{self.text}\n"
            f"METADATA=\n"
             "{\n"
            f"'document_metadata': {self.doc_meta},\n"
            f"'chunk_metadata': {self.chunk_meta},\n"
             "}\n"
            f")"
        )

    # @classmethod
    # def from_dict(data: dict):
    #     chunk = Chunk(
    #         text=data.get("text", ""),
    #         doc_name=data.get("doc_name", ""),
    #         doc_type=data.get("doc_type", ""),
    #         doc_id=data.get("doc_id", ""),
    #         chunk_id=data.get("chunk_id", ""),
    #         meta=data.get("meta", {}),
    #     )
    #     chunk.tokens = data.get("tokens", 0)
    #     chunk.score = data.get("score", 0.0)
    #     return chunk
    
class Document:
    def __init__(
        self,
        text: str="",
        type: str="",
        name: str="",
        path: str="",
        link: str="",
        timestamp: str="",
        reader: str="",
        meta: dict=None,
    ):
        if meta is None:
            meta = {}
        
        self.text = text
        self.type = type
        self.name = name
        self.path = path
        self.link = link
        self.timestamp = timestamp
        self.reader = reader
        self.meta = meta
        self.chunks: list[Chunk] = []
    
    # def to_dict(self) -> dict:
    #     doc_dict = {
    #         "text": self.text,
    #         "type": self.type,
    #         "name": self.name,
    #         "path": self.path,
    #         "link": self.link,
    #         "timestamp": self.timestamp,
    #         "reader": self.reader,
    #         "meta": self.meta,
    #         "chunks": [chunk.to_dict() for chunk in self.chunks],
    #     }
    #     return doc_dict
    
    # @staticmethod
    # def from_dict(data: dict):
    #     document = Document(
    #         text=data.get("text", ""),
    #         type=data.get("type", ""),
    #         name=data.get("name", ""),
    #         path=data.get("path", ""),
    #         link=data.get("link", ""),
    #         timestamp=data.get("timestamp", ""),
    #         reader=data.get("reader", ""),
    #         meta=data.get("meta", {}),
    #     )
    #     document.chunks = [Chunk.from_dict(chunk) for chunk in data.get("chunks", [])]
    #     return document
    

class FileData(BaseModel):
    filename: str
    extension: str
    content: str


class CombinedChunks:
    def __init__(self, chunks: dict[str, Chunk]=None, doc_id="", doc_name="", doc_type="base"):
        self.chunks = chunks if chunks is not None else {}
        self.doc_type = doc_type
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.score = 0.0
        self.url = None
        self.base_doc_id = None
        self.num_chunks = 0