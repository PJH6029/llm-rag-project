from typing import Iterator, Iterable

from langchain_text_splitters import TextSplitter

from rag.type import *

def lazy_chunk_with(
    text_splitter: TextSplitter,
    chunks: Iterable[Chunk],
    *,
    with_parent_mark: bool = False,
    parent_id_key: str = "parent_id",
) -> Iterator[Chunk]:
    for chunk in chunks:
        text = chunk.text
        splitted_texts = text_splitter.split_text(text) 

        for i, splitted_text in enumerate(splitted_texts):
            sub_chunk = Chunk(
                text=splitted_text,
                doc_id=chunk.doc_id,
                chunk_id=f"{chunk.chunk_id}-{i}",
                doc_meta={**chunk.doc_meta},
                chunk_meta={**chunk.chunk_meta},
            )
            if with_parent_mark:
                sub_chunk.chunk_meta[parent_id_key] = chunk.chunk_id
            yield sub_chunk

def chunk_with(
    text_splitter: TextSplitter,
    chunks: Iterable[Chunk],
    *,
    with_parent_mark: bool = False,
) -> list[Chunk]:
    return list(lazy_chunk_with(text_splitter, chunks, with_parent_mark=with_parent_mark))
