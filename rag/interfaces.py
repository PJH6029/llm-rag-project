import tiktoken
from wasabi import msg
# from typing import AsyncIterator
from typing import Iterator

from rag.types import *
from langchain.prompts import ChatPromptTemplate


class Component:
    pass


class Reader(Component):
    def __init__(self) -> None:
        super().__init__()

    def load(self, fileData: list[FileData]) -> list[Document]:
        raise NotImplementedError()
    

class Chunker(Component):
    def __init__(self) -> None:
        super().__init__()

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        raise NotImplementedError()


class Embedder(Component):
    def __init__(self) -> None:
        super().__init__()

    def embed(self, documents: list[Document]) -> bool:
        raise NotImplementedError() 


class Retriever(Component):
    def __init__(self) -> None:
        super().__init__()

    def retrieve(self, query: str, embedder: Embedder, top_k: int=5) -> dict[str, list[Chunk]]:
        raise NotImplementedError()


class Generator(Component):
    def __init__(self) -> None:
        super().__init__()
        self.prompt = None
        self.model_name = ""
        self.context_window = 0

    def generate(self, queries: list[str], context: str, history: list[dict]) -> list[str]:
        raise NotImplementedError()
    
    # TODO async
    def generate_stream(self, queries: list[str], context: dict[str, str], history: list[dict]) -> Iterator[str]:
        raise NotImplementedError()


class Revisor(Component):
    def __init__(self) -> None:
        super().__init__()
        self.prompt = None
        self.model_name = ""
        self.context_window = 0

    def revise(self, queries: list[str], history: list[dict], revise_prompt: ChatPromptTemplate=None) -> list[str]:
        raise NotImplementedError()
