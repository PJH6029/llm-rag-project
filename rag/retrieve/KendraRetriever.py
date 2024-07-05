from dotenv import load_dotenv
import os
from wasabi import msg

from rag.interfaces import Embedder, Retriever
from rag.types import *

from langchain_community.retrievers import AmazonKendraRetriever
load_dotenv()

class KendraRetriever(Retriever):
    def __init__(self) -> None:
        super().__init__()
        self.top_k = 5

    def retrieve(self, queries: list[str], embedder: Embedder) -> tuple[list[Chunk], str]:
        kendraRetriever = AmazonKendraRetriever(
            index_id=os.environ["KENDRA_INDEX_ID"],
            region_name=os.environ["AWS_REGION"],
            top_k=self.top_k
        )
        msg.info("Kendra Retriever initialized. Retrieving chunks...")
        chunks = []
        for query in queries:
            retrieved_chunks = kendraRetriever.invoke(query)
            for chunk_raw in retrieved_chunks:
                doc_type = chunk_raw.metadata["title"].split(".")[-1] # TODO
                meta = self.process_metadata(chunk_raw.metadata)
                chunk = Chunk(
                    text=chunk_raw.metadata["excerpt"],
                    doc_name=chunk_raw.metadata["title"],
                    doc_type=doc_type,
                    doc_id=chunk_raw.metadata["document_id"],
                    chunk_id=chunk_raw.metadata["result_id"],
                    meta=meta,
                )
                score = 5 if chunk_raw.metadata["score"] == "HIGH" else 3 if chunk_raw.metadata["score"] == "MEDIUM" else 1
                chunk.score = score

                chunks.append(chunk)
        
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        context = self.combine_context(chunks)
        return sorted_chunks, context
    
    def process_metadata(self, metadata: dict) -> dict:
        processed_metadata = {
            "excerpt_page_number": metadata.get("document_attributes", {}).get("_excerpt_page_number", ""),
        }
        return processed_metadata
    
    def combine_chunks(self, chunks: list[Chunk]) -> dict:
        docs = {}
        for chunk in chunks:
            if chunk.doc_id not in docs:
                docs[chunk.doc_id] = {"score": 0.0, "chunks": {}, "doc_name": chunk.doc_name}
            docs[chunk.doc_id]["score"] += float(chunk.score)
            docs[chunk.doc_id]["chunks"][chunk.chunk_id] = chunk
        # sort docs. preventing lost in the middle problem (see https://arxiv.org/abs/2307.03172)
        docs = dict(sorted(docs.items(), key=lambda x: x[1]["score"], reverse=True))
        return docs
    
    def combine_context(self, chunks: list[Chunk]) -> str:
        docs = self.combine_chunks(chunks)
        context = ""
        
        docs = dict(list(docs.items())[:self.top_k])

        for doc_id in docs:
            # sort chunk by scores
            sorted_chunks = list(sorted(docs[doc_id]["chunks"].values(), key=lambda chunk: chunk.score, reverse=True))

            context += f"--- Document: {docs[doc_id]['doc_name']} ---\n\n"

            for chunk in sorted_chunks:
                context += f"{chunk.to_detailed_str()}\n\n"
            
        return context
    