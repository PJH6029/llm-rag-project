from typing import Optional
from wasabi import msg

from rag.managers.base import BasePipelineManager
from rag.type import *
from rag.model.retrievers import *
from rag.model import embeddings


class RetrieverManager(BasePipelineManager):
    DEFAULT_TOP_K = 5
    FALLBACK_RETRIEVER = "default"
    
    def __init__(self) -> None:
        super().__init__()
        self.embeddings_name: Optional[str] = None
        self.embeddings: Optional[Embeddings] = None
        self.top_k = self.DEFAULT_TOP_K
        
        self.retrievers: dict[str, BaseRAGRetriever] = {
            "default": BaseRAGRetriever,
            "kendra": KendraRetriever,
            "knowledge-base-oss": KnowledgeBaseOpenSearchRetriever,
            "knowledge-base-pinecone": KnowledgeBasePineconeRetriever,
            "pinecone": PineconeRetriever,
            "ensemble": EnsembleRetriever,
        }
        self.selected_retriever_name: str = self.FALLBACK_RETRIEVER
        self.weights: list[float] = []
        self.selected_retriever: Optional[BaseRAGRetriever] = None


    def set_config(self, config: dict):
        self.embeddings_name = config.get("embedding")
        self.embeddings = embeddings.get_model(self.embeddings_name) if self.embeddings_name is not None else None
        msg.info(f"Setting EMBEDDINGS to {self.embeddings_name}")

        self.top_k = config.get("top_k", self.DEFAULT_TOP_K)
        
        selected_retriever_names = config.get("retriever", self.FALLBACK_RETRIEVER)
        if len(selected_retriever_names) == 1:
            self.selected_retriever_name = selected_retriever_names[0]
            self.selected_retriever = self.retrievers.get(self.selected_retriever_name)(
                top_k=self.top_k,
                embeddings=self.embeddings,
            )
            msg.info(f"Setting RETRIEVER to {self.selected_retriever_name}")
        else:
            # ensemble retriever
            if "weights" not in config:
                msg.warn("Weights not found for ensemble retriever. Using equal weights.")
            
            self.selected_retriever_name = "ensemble"
            self.selected_retriever = self.retrievers.get(self.selected_retriever_name)(
                retrievers=[
                    self.retrievers.get(retriever_name)(
                        top_k=self.top_k,
                        embeddings=self.embeddings,
                    ) for retriever_name in selected_retriever_names
                ],
                weights=config.get("weights"),
            )
            msg.info(f"Setting RETRIEVER to ensemble of {selected_retriever_names}")

        self.post_retrieval_config = config.get("post_retrieval", {}) # TODO
        
        
    def retrieve(self, queries: list[str], filter: dict=None) -> list[Chunk]:
        """_summary_

        Args:
            queries (list[str]): list of queries to retrieve
            filter (dict, optional): Use syntax of AWS Knowledge base filter (https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html). Defaults to None.

        Returns:
            list[Chunk]: list of retrieved chunks
        """
        retriever = self.selected_retriever
        if retriever is None:
            msg.warn(f"Retriever {self.selected_retriever_name} not found. Skipping retrieval.")
            return []

        formulated_filter = FilterUtil.from_dict(filter)
        retrieved_chunks = retriever.retrieve(queries, filter=formulated_filter)
        return retrieved_chunks
    
    def rerank(self, chunks: list[Chunk]) -> list[Chunk]:
        if self.selected_retriever_name == "ensemble":
            # they are already ranked
            return chunks[:self.top_k]
        
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        return sorted_chunks[:self.top_k] # TODO
