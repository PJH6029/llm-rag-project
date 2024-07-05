import tiktoken
from wasabi import msg
# from typing import AsyncIterator
from typing import Iterator

from rag.types import *



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

    def retrieve(self, queries: list[str], embedder: Embedder) -> tuple[list[Chunk], str]:
        raise NotImplementedError()
    
    def cutoff_text(self, text: str, content_length: int) -> str:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        encoded_tokens = encoding.encode(text, disallowed_special=())

        if len(encoded_tokens) > content_length:
            truncated_tokens = encoded_tokens[:content_length]
            truncated_text = encoding.decode(truncated_tokens)
            msg.info(f"Truncated text from {len(encoded_tokens)} tokens to {len(truncated_tokens)} tokens")
            return truncated_text
        else:
            msg.info(f"Text has {len(encoded_tokens)} tokens, not truncating")
            return text


class Generator(Component):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, queries: list[str], context: str, history_str: str="No history") -> list[str]:
        raise NotImplementedError()
    
    # TODO async
    def generate_stream(self, queries: list[str], context: str, history_str: str="No history") -> Iterator[str]:
        raise NotImplementedError()