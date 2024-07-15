import os
from wasabi import msg
from pprint import pprint

from rag.interfaces import Embedder, Retriever
from rag.types import *

from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
class RetrievalConfig:
    def __init__(self, config: dict) -> None:
        self._config = config
    
    def dict(self):
        return self._config

class KnowledgeBaseRetriever(Retriever):
    def __init__(self) -> None:
        super().__init__()

    def retrieve(self, query: str, embedder: Embedder, top_k: int=5, category="base", base_doc_id=None) -> list[Chunk]:
        knowledgeBaseRetriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=os.environ["KNOWLEDGE_BASE_ID"],
            region_name=os.environ["AWS_REGION"],
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                }
            }
        )
        # msg.info("KnowledgeBaseRetriever initialized.")

        if category == "base":
            return self._retrieve_base(knowledgeBaseRetriever, query, top_k)
        else:
            return self._retrieve_additional(knowledgeBaseRetriever, query, top_k, base_doc_id)

    def _retrieve_base(self, retriever: AmazonKnowledgeBasesRetriever, query: str, top_k: int) -> list[Chunk]:
        retrieval_config = {
            "vectorSearchConfiguration": {
                "filter": {
                    "equals": {
                        "key": "category",
                        "value": "base",
                    }
                },
                "numberOfResults": top_k,
            }
        }
        retriever.retrieval_config = RetrievalConfig(retrieval_config)
        retrieved_base_chunks_raw = retriever.invoke(query)
        retrieved_base_chunks = [self.process_chunk(chunk_raw, "base") for chunk_raw in retrieved_base_chunks_raw]
        return retrieved_base_chunks

    def _retrieve_additional(self, retriever: AmazonKnowledgeBasesRetriever, query: str, top_k: int, base_doc_id: str) -> list[Chunk]:
        retrieval_config = {
            "vectorSearchConfiguration": {
                "filter": {
                    "equals": {
                        "key": "category",
                        "value": "additional",
                    },
                    "equals": {
                        "key": "base-doc-id",
                        "value": base_doc_id,
                    }
                },
                "numberOfResults": top_k,
            }
        }
        retriever.retrieval_config = RetrievalConfig(retrieval_config)
        retrieved_additional_chunks_raw = retriever.invoke(query)
        retrieved_chunks = [self.process_chunk(chunk_raw, "additional") for chunk_raw in retrieved_additional_chunks_raw]
        return retrieved_chunks

    def process_chunk(self, chunk_raw: dict, doc_type: str) -> Chunk:
        doc_meta, chunk_meta = self.process_metadata(chunk_raw.metadata, doc_type)
        chunk = Chunk(
            text=chunk_raw.page_content,
            doc_id=chunk_raw.metadata['location']['s3Location']['uri'],
            chunk_id=chunk_raw.metadata['source_metadata']['x-amz-bedrock-kb-chunk-id'],
            doc_meta=doc_meta,
            chunk_meta=chunk_meta
        )
        chunk.score = chunk_raw.metadata['score']
        return chunk
    
    def process_metadata(self, metadata: dict, doc_type: str) -> tuple[dict, dict]:
        doc_meta = {
            'doc_name': metadata['location']['s3Location']['uri'].split('/')[-1],
            'doc_type': metadata['source_metadata']['category'], # TODO it should equals to doc_type
            'version': metadata['source_metadata']['version'],
            'uri': metadata['location']['s3Location']['uri'],
        }

        chunk_meta = {
            'score': metadata['score'],
            'excerpt_page_number': -1, # TODO
        }
        if doc_type == "additional":
            chunk_meta['base_doc_id'] = metadata['source_metadata']['base-doc-id']
        return doc_meta, chunk_meta