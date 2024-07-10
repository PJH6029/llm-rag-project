from dotenv import load_dotenv
import os
from pprint import pprint
from wasabi import msg

from rag.interfaces import Embedder, Retriever
from rag.types import *

from langchain_community.retrievers import AmazonKendraRetriever
load_dotenv()

class KendraRetriever(Retriever):
    def __init__(self) -> None:
        super().__init__()

    def retrieve(self, query: list[str], embedder: Embedder, top_k: int=5) -> dict[str, list[Chunk]]:
        kendraRetriever = AmazonKendraRetriever(
            index_id=os.environ["KENDRA_INDEX_ID"],
            region_name=os.environ["AWS_REGION"],
            top_k=top_k,
        )
        msg.info("KendraRetriever initialized.")

        # base-context-retrieval
        msg.info("Retrieving base context...")
        attribute_filter = {
            "EqualsTo": {
                "Key": "_category",
                "Value": {
                    "StringValue": "base",
                },
            }
        }
        kendraRetriever.attribute_filter = attribute_filter
        retrieved_base_chunks_raw = kendraRetriever.invoke(query)
        retrieved_base_chunks = [self.process_chunk(chunk_raw, "base") for chunk_raw in retrieved_base_chunks_raw]
        msg.good(f"Retrieved {len(retrieved_base_chunks)} base chunks.")

        base_doc_ids = set([chunk.doc_id for chunk in retrieved_base_chunks])
        msg.info(f"Retrieved base doc ids: {base_doc_ids}")

        # additional-context-retrieval
        retrieved_additional_chunks = []
        for base_doc_id in base_doc_ids:
            msg.info(f"Retrieving additional context for base doc id: {base_doc_id}...")
            attribute_filter = {
                "EqualsTo": {
                    "Key": "_category",
                    "Value": {
                        "StringValue": "additional",
                    },
                },
                "EqualsTo": {
                    "Key": "base-doc-id",
                    "Value": {
                        "StringValue": base_doc_id,
                    }
                }
            }
            kendraRetriever.attribute_filter = attribute_filter
            retrieved_additional_chunks_raw = kendraRetriever.invoke(query)
            retrieved_chunks = [self.process_chunk(chunk_raw, "additional") for chunk_raw in retrieved_additional_chunks_raw]
            retrieved_additional_chunks.extend(retrieved_chunks)
            msg.good(f"Retrieved {len(retrieved_chunks)} additional chunks")
            
        return {"base": retrieved_base_chunks, "additional": retrieved_additional_chunks}

    def process_chunk(self, chunk_raw: dict, doc_type: str) -> Chunk:
        doc_meta, chunk_meta = self.process_metadata(chunk_raw.metadata, doc_type)
        chunk = Chunk(
            text=chunk_raw.metadata["excerpt"],
            doc_id=chunk_raw.metadata["document_id"],
            chunk_id=chunk_raw.metadata["result_id"],
            doc_meta=doc_meta,
            chunk_meta=chunk_meta,
        )
        score = 1 if chunk.chunk_meta["score"] == "HIGH" else 0.5 if chunk.chunk_meta["score"] == "MEDIUM" else 1
        chunk.score = score
        return chunk
    
    def process_metadata(self, metadata: dict, doc_type: str) -> dict:
        doc_meta = {
            "doc_name": metadata.get("title", ""),
            "doc_type": doc_type, # base or additional
            "version": metadata.get("document_attributes", {}).get("version", ""),
            "uri": metadata.get("document_attributes", {}).get("_source_uri", ""),
        }

        chunk_meta = {
            "score": metadata.get("score", ""),
            "excerpt_page_number": metadata.get("document_attributes", {}).get("_excerpt_page_number", ""),
        }
        if doc_type == "additional":
            chunk_meta["base_doc_id"] = metadata.get("document_attributes", {}).get("base-doc-id", "")
        return doc_meta, chunk_meta
