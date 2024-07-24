from typing import Optional, Type, Callable
from wasabi import msg

from rag.managers.base import BasePipelineManager
from rag.type import *
from rag.model.retrievers import *
from rag.model import embeddings
from rag import util


class RetrieverManager(BasePipelineManager):
    DEFAULT_TOP_K = 5
    FALLBACK_RETRIEVER_NAME = "default"
    
    def __init__(self) -> None:
        super().__init__()
        self.embeddings_name: Optional[str] = None
        self.embeddings: Optional[Embeddings] = None
        self.top_k = self.DEFAULT_TOP_K
        
        self.retrievers: dict[str, Type[BaseRAGRetriever]] = {
            "default": BaseRAGRetriever,
            "kendra": KendraRetriever,
            "knowledge-base-oss": KnowledgeBaseOpenSearchRetriever,
            "knowledge-base-pinecone": KnowledgeBasePineconeRetriever,
            "pinecone": PineconeRetriever,
        }

        self.selected_retriever_names: list[str] = []
        self.use_context_hierarchy: bool = False
        self.weights: list[float] = []
        
        self.post_retrieval_config: dict = {}
        
        self.selected_retriever: Optional[BaseRAGRetriever] = None


    def set_config(self, config: dict):
        self.embeddings_name = config.get("embedding")
        
        if self.embeddings_name is not None:
            self.embeddings = embeddings.get_model(self.embeddings_name)
            msg.info(f"Setting EMBEDDINGS to {self.embeddings_name}")
        else:
            self.embeddings = None
            msg.warn("EMBEDDINGS not configured. Setting to None.")

        self.top_k = config.get("top_k", self.DEFAULT_TOP_K)
        
        self.selected_retriever_names = config.get("retriever", [self.FALLBACK_RETRIEVER_NAME])
        self.weights = config.get("weights", [])
        self.use_context_hierarchy = config.get("context-hierarchy", False)

        self.post_retrieval_config = config.get("post_retrieval", {}) # TODO
        
        init_params = {
            "top_k": self.top_k,
            "embeddings": self.embeddings,
        }
        init_params = util.remove_falsy(init_params)
        self.init_retriever(init_params)
    
        
    def _ensemble_lambda(self) -> Callable[..., BaseRAGRetriever]:
        if len(self.selected_retriever_names) == 1:
            msg.info(f"Setting RETRIEVER to {self.selected_retriever_names[0]}")
            def retriever_initiator(**init_params):
                return self.retrievers[self.selected_retriever_names[0]](
                    **init_params
                )
        else:
            # ensemble retriever
            if self.weights is None:
                msg.warn("Weights not provided. Using equal weights.")
                self.weights = [1.0] * len(self.selected_retriever_names)
            
            msg.info(f"Setting RETRIEVER to ensemble of {self.selected_retriever_names}")
            def retriever_initiator(**init_params):
                return EnsembleRetriever(
                    retrievers=[
                        self.retrievers[retriever_name](
                            **init_params
                        ) for retriever_name in self.selected_retriever_names
                    ],
                    weights=self.weights,
                    **init_params
                )
        return retriever_initiator

    
    def init_retriever(self, init_params: dict) -> BaseRAGRetriever:
        try:
            ensemble_lambda = self._ensemble_lambda()
            
            def init_with_hierarchy(**kwargs):
                retriever_builder = ensemble_lambda
                return HierarchicalRetriever.from_retriever(retriever_builder, **kwargs)
            
            msg.info(f"Use Hierarchical Retriever: {self.use_context_hierarchy}")
            retriever_factory = init_with_hierarchy if self.use_context_hierarchy else ensemble_lambda
            
            retriever = retriever_factory(**init_params)
        except KeyError as e:
            key = e.args[0]
            msg.warn(f"Retriever {key} not found. Using fallback retriever.")
            
            retriever = self.retrievers[self.FALLBACK_RETRIEVER_NAME](**init_params)
        except ValueError as e:
            msg.warn(f"Error initializing retriever: {e}. Using fallback retriever.")
            retriever = self.retrievers[self.FALLBACK_RETRIEVER_NAME](**init_params)
        finally:
            self.selected_retriever = retriever
        
        
    def retrieve(self, queries: list[str], filter: dict | None=None) -> list[Chunk]:
        """Retrieve chunks from selected retriever

        Args:
            queries (list[str]): list of queries to retrieve
            filter (dict, optional): Use syntax of AWS Knowledge base filter (https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html). Defaults to None.

        Returns:
            list[Chunk]: list of retrieved chunks
        """
        retriever = self.selected_retriever
        if retriever is None:
            msg.warn(f"Retriever is None. Skipping retrieval.")
            return []

        formulated_filter = FilterUtil.from_dict(filter)
        retrieved_chunks = retriever.retrieve(queries, filter=formulated_filter)
        return retrieved_chunks

