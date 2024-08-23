from typing import Optional, Iterable
from wasabi import msg
import time
import itertools
import random
import os

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from rag.component.ingestor.base import BaseRAGIngestor
from rag.component.ingestor.PineconeVectorstoreIngestor import PineconeVectorstoreIngestor
from rag.type import Chunk
from rag.component import chunker, embeddings, llm
from rag.component.ingestor import prompt
from rag.util import time_logger
from rag.config import IngestionConfig

class ExponentialBackoff:
    def __init__(self, initial_delay=1, max_delay=60, factor=2, jitter=True):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.factor = factor
        self.jitter = jitter

    def get_delay(self, attempt):
        delay = self.initial_delay * (self.factor ** attempt)
        if self.jitter:
            delay += random.uniform(0, 1)  # Add some randomness to avoid thundering herd
        return min(delay, self.max_delay)

def retry_with_exponential_backoff(func, max_retries=5, backoff=ExponentialBackoff(initial_delay=30, max_delay=120)):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                delay = backoff.get_delay(attempt)
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Attempt {attempt + 1} failed: {e}. No more retries.")
                raise

def generate_chunks_with_retries(chunk_generator: "ChunkGenerator", chunks: list[Chunk]) -> list[Chunk]:
    def generate_chunks():
        return chunk_generator.generate(chunks)
    
    return retry_with_exponential_backoff(generate_chunks)

class ChunkGenerator:
    def __init__(
        self, 
        llm_model_name: str = "gpt-4o-mini",
        parent_id_key: str = "parent_id",
        lang: str = "English",
    ) -> None:
        self.llm_model_name = llm_model_name
        self.parent_id_key = parent_id_key
        self.lang = lang
        
    def generate(self, chunks: list[Chunk]) -> list[Chunk]:
        parallel_chain = RunnableParallel(
            split=RunnableLambda(self._split_runnable),
            summarize=RunnableLambda(self._summarize_runnable),
            hypothetical_query=RunnableLambda(self._hypothetical_query_runnable),
        )
        
        result = parallel_chain.invoke({"chunks": chunks})
        return list(itertools.chain(*result.values()))

    def _split_runnable(self, _dict):
        return self._split(_dict["chunks"])
    
    def _summarize_runnable(self, _dict):
        return self._summarize(_dict["chunks"])

    def _hypothetical_query_runnable(self, _dict):
        return self._hypothetical_query(_dict["chunks"])

    def _split(
        self, 
        chunks: list[Chunk],
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        with_parent_mark: bool = True,
    ) -> list[Chunk]:
        with time_logger(
            lambda: f"Splitting {len(chunks)} chunks...",
            lambda: f"Splitting completed. {len(splitted_chunks)} chunks generated"
        ):
            # TODO temporal sub-chunking
            splitted_chunks = chunker.chunk_with(
                RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
                chunks,
                with_parent_mark=with_parent_mark,
            )
            return splitted_chunks
    
    def _summarize(self, chunks: list[Chunk]) -> list[Chunk]:
        with time_logger(
            lambda: f"Summarizing {len(chunks)} chunks...",
            lambda: f"Summarizing completed. {len(new_chunks)} chunks generated"
        ):
            chunks = list(chunks)
        
            chain = prompt.summarize_prompt.partial(lang=self.lang) | llm.get_model(self.llm_model_name) | StrOutputParser()
            summaries = chain.batch([{"text": chunk.text, "doc_meta": chunk.doc_meta, "chunk_meta": chunk.chunk_meta} for chunk in chunks])
            new_chunks = []
            for chunk, summary in zip(chunks, summaries):
                new_chunk_id = f"{chunk.chunk_id}-summary"
                new_chunk = Chunk(
                    text=summary,
                    doc_id=chunk.doc_id,
                    chunk_id=new_chunk_id,
                    doc_meta={**chunk.doc_meta},
                    chunk_meta={**chunk.chunk_meta},
                )
                new_chunk.chunk_meta[self.parent_id_key] = chunk.chunk_id
                new_chunks.append(new_chunk)
            return new_chunks

    def _hypothetical_query(self, chunks: list[Chunk]) -> list[Chunk]:
        # reverse hyde
        with time_logger(
            lambda: f"Generating hypothetical queries for {len(chunks)} chunks...",
            lambda: f"Generating hypothetical queries completed. {len(new_chunks)} chunks generated"
        ):
            chunks = list(chunks)
        
            chain = prompt.hypothetical_queries_prompt.partial(lang=self.lang) | llm.get_model(self.llm_model_name) | StrOutputParser()
            queries_list = chain.batch([{"n": 3 , "text": chunk.text, "doc_meta": chunk.doc_meta, "chunk_meta": chunk.chunk_meta} for chunk in chunks])
            
            new_chunks = []
            for chunk, queries in zip(chunks, queries_list):
                queries = [q for q in queries.split("\n") if q]

                _new_chunks = []
                for i, query in enumerate(queries):
                    new_chunk_id = f"{chunk.chunk_id}-query-{i}"
                    new_chunk = Chunk(
                        text=query,
                        doc_id=chunk.doc_id,
                        chunk_id=new_chunk_id,
                        doc_meta=chunk.doc_meta,
                        chunk_meta={**chunk.chunk_meta, "chunk_id": new_chunk_id}
                    )
                    new_chunk.chunk_meta[self.parent_id_key] = chunk.chunk_id
                    _new_chunks.append(new_chunk)
                new_chunks.extend(_new_chunks)
            return new_chunks

class PineconeMultiVectorIngestor(BaseRAGIngestor):
    CHILD_INGESTION_CNT: int = 0
    INGEST_FROM_SCRATCH: bool = True # Set to False to skip already ingested chunks
    
    def __init__(
        self,
        embeddings: Embeddings,
        parent_namespace: str,
        child_namespace: str,
        source_lang: str = "English",
        ingestion_log_path: Optional[str] = "ingestor_logs.txt",
    ) -> None:
        super().__init__()
        self.parent_ingestor = PineconeVectorstoreIngestor(
            embeddings=embeddings,
            namespace=parent_namespace
        )
        self.child_ingestor = PineconeVectorstoreIngestor(
            embeddings=embeddings,
            namespace=child_namespace
        )
        self._embeddings = embeddings
        self._parent_namespace = parent_namespace
        self._child_namespace = child_namespace
        self._source_lang = source_lang
        
        self.ingestion_log: dict[str, set] = {}
        if os.path.exists(ingestion_log_path):
            with open(ingestion_log_path, "r") as f:
                for line in f:
                    doc_id, page = line.strip().split(",")
                    if doc_id not in self.ingestion_log:
                        self.ingestion_log[doc_id] = set()
                    self.ingestion_log[doc_id].add(int(page))
    
    def ingest(self, chunks: list[Chunk]) -> int:
        """In addition to the parent ingestion, this ingestor also chunks further and ingests the chunks into the child namespace.
 
        Args:
            chunks (list[Chunk]): List of chunks to ingest

        Returns:
            bool: True if ingestion is successful
        """
        
        if not PineconeMultiVectorIngestor.INGEST_FROM_SCRATCH:
            # Filter out already ingested chunks
            filtered_chunks = [c for c in chunks if c.doc_id not in self.ingestion_log or c.chunk_meta.get("page", -1) not in self.ingestion_log[c.doc_id]]
            if len(filtered_chunks) != len(chunks):
                msg.warn(f"Skipping {len(chunks) - len(filtered_chunks)} already ingested chunks")
            chunks = filtered_chunks
        
        if len(chunks) == 0:
            msg.warn("No new chunks to ingest")
            return 0
        
        msg.info(f"Ingesting {len(chunks)} chunks")
        chunk_generator = ChunkGenerator(lang=self._source_lang)
        children_chunks = generate_chunks_with_retries(chunk_generator, chunks)
        
        parent_ingestion_cnt = self.parent_ingestor.ingest(chunks)
        child_ingestion_cnt = self.child_ingestor.ingest(children_chunks)
        PineconeMultiVectorIngestor.CHILD_INGESTION_CNT += child_ingestion_cnt

        num_chunk_ids = len(set([c.chunk_id for c in chunks]))
        if len(chunks) != num_chunk_ids:
            msg.warn(f"Duplicate chunk ids found in parent chunks. # of chunks: {len(chunks)}, # of unique chunk ids: {num_chunk_ids}")
            return 0
        
        num_child_chunk_ids = len(set([c.chunk_id for c in children_chunks]))
        if len(children_chunks) != num_child_chunk_ids:
            msg.warn(f"Duplicate chunk ids found in child chunks. # of chunks: {len(children_chunks)}, # of unique chunk ids: {num_child_chunk_ids}")
            return 0
    
        msg.good(f"{len(chunks)} chunks ingested into parent namespace `{self.parent_ingestor.namespace}`")
        msg.good(f"{len(children_chunks)} chunks ingested into child namespace `{self.child_ingestor.namespace}`")
        
        # print(f"Ingested chunks:")
        logs = ""
        for c in chunks:
            log = f"{c.doc_id},{c.chunk_meta.get('page', -1)}\n"
            # print(log, end="")
            logs += log
            
            if c.doc_id not in self.ingestion_log:
                self.ingestion_log[c.doc_id] = set()
            self.ingestion_log[c.doc_id].add(c.chunk_meta.get("page", -1))
        with open("ingestor_logs.txt", "a") as f:
            f.write(logs)
        
        return parent_ingestion_cnt
    
    @classmethod
    def from_config(cls, config: IngestionConfig) -> "PineconeMultiVectorIngestor":
        embeddings_name = config.embeddings
        parent_namespace = config.namespace
        child_namespace = config.sub_namespace
        
        embeddings_model = embeddings.get_model(embeddings_name)
        
        source_lang = config.global_.lang.source
        
        return cls(
            embeddings=embeddings_model,
            parent_namespace=parent_namespace,
            child_namespace=child_namespace,
            source_lang=source_lang,
        )
