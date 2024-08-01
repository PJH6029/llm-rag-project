import os
from typing import Optional
from pinecone import Pinecone
import uuid
import pprint
from wasabi import msg

from langchain_pinecone import PineconeVectorStore as PVS
from langchain_core.documents import Document

from rag.component.vectorstore.base import BaseRAGVectorstore
from rag.type import Chunk, Embeddings
from rag import util

class PineconeVectorstore(BaseRAGVectorstore):
    def __init__(
        self, 
        embeddings: Embeddings | None = None,
        namespace: Optional[str] = None,
        text_key: Optional[str] = "text",
        **kwargs
    ) -> None:
        super().__init__(embeddings)
        self.vectorstore = PVS.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=namespace,
        )
        self.namespace = namespace
        self._text_key = text_key
    
    def _set_env(self):
        self.api_key = os.environ["PINECONE_API_KEY"]
        self.index_name = os.environ["PINECONE_INDEX_NAME"]
        
    def ingest(self, chunks: list[Chunk]) -> int:
        docs = [chunk.to_document() for chunk in chunks]
        
        for doc in docs:
            _meta = util.flatten_dict(doc.metadata)
            
            if not _meta.get("chunk_id"):
                _meta["chunk_id"] = str(uuid.uuid4())
            if not _meta.get("doc_id"):
                if not _meta.get("source"):
                    raise ValueError("doc_id or source must be provided in metadata")
                _meta["doc_id"] = _meta["source"]

            doc.metadata = _meta
        
        ids = [doc.metadata["chunk_id"] for doc in docs]
        
        # print(self.namespace, len(docs), ids)
        self.vectorstore.add_documents(
            documents=docs,
            ids=ids,
            namespace=self.namespace,
        )
        return len(docs)

    def query(self, query: str, top_k: int = 5, filter: dict | None=None) -> list[Chunk]:
        result = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
            namespace=self.namespace,  
            filter=filter 
        )
        
        if result:
            retrieved_chunks_raw, scores = zip(*result)
        else:
            retrieved_chunks_raw, scores = [], []
        
        chunks = []
        for chunk_raw, score in zip(retrieved_chunks_raw, scores):
            restored_metadata = util.deflatten_dict(chunk_raw.metadata)
            chunks.append(Chunk(
                text=chunk_raw.page_content,
                doc_id=util.MetadataSearch.search_doc_id(restored_metadata),
                chunk_id=util.MetadataSearch.search_chunk_id(restored_metadata),
                doc_meta=restored_metadata["doc_meta"],
                chunk_meta={**restored_metadata["chunk_meta"], "score": score},
                score=score,
                source_retriever=self.__class__.__name__,
            ))
        return chunks
        
    
    def fetch(self, ids: list[str]) -> list[Chunk]:
        retrieved_chunks_raw = self._fetch_docs(ids)
        chunks = []
        for chunk_raw in retrieved_chunks_raw:
            restored_metadata = util.deflatten_dict(chunk_raw.metadata)
            chunks.append(Chunk(
                text=chunk_raw.page_content,
                doc_id=restored_metadata["doc_id"],
                chunk_id=restored_metadata["chunk_id"],
                doc_meta=restored_metadata["doc_meta"],
                chunk_meta=restored_metadata["chunk_meta"],
                source_retriever=self.__class__.__name__,
            ))
        return chunks
        
    def _fetch_docs(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        
        result = self.vectorstore._index.fetch(
            ids=ids,
            namespace=self.namespace,   
        )
        
        docs = []
        for res in result["vectors"].values():
            metadata = res["metadata"]
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                docs.append(Document(page_content=text, metadata=metadata))
            else:
                msg.warn(
                    f"Found document with no `{self._text_key}` key. Skipping."
                )
        return docs
    
    def fetch_docs(self, ids: list[str]) -> list[Document]:
        return self._fetch_docs(ids)