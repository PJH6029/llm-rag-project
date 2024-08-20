from wasabi import msg
from typing import Generator, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from rag.component import llm, prompt
from rag.managers.base import BasePipelineManager
from rag.type import Chunk, ChatLog
from rag import util
from rag.config import GenerationConfig

class GeneratorManager(BasePipelineManager):
    def __init__(self) -> None:
        super().__init__()
        self.generator_name = None
        self.prompt = None
        
    
    def set_config(self, config: GenerationConfig):
        self.generator_name = config.model
        use_context_hierarchy = config.global_.context_hierarchy
        self.user_lang = config.global_.lang.user
        
        if use_context_hierarchy:
            _prompt = prompt.generation_with_hierarchy_prompt
        else:
            _prompt = prompt.generation_without_hierarchy_prompt
        self.prompt = _prompt.partial(lang=self.user_lang)
        
        msg.info(f"Setting GENERATOR to {self.generator_name}")

    def generate_stream(
        self, query: str, history: str="", context: str=""
    ) -> Generator[str, None, None]:
        chain = self._get_chain()
        if chain is None:
            return
        
        yield from chain.stream({"query": query, "context": context, "history": history})
    
    def generate(
        self, query: str, history: str="", context: str=""
    ) -> str:
        chain = self._get_chain()
        if chain is None:
            return ""
        
        return chain.invoke({"query": query, "context": context, "history": history})
    
    def _get_chain(self) -> Optional[Runnable]:
        if self.generator_name is None:
            msg.warn("Generator not set. Skipping generation.")
            return None
        
        generator = llm.get_model(self.generator_name, temperature=0.0)
        if generator is None:
            msg.warn(f"Generator {self.generator_name} not found. Skipping generation.")
            return None

        chain = self.prompt | generator | StrOutputParser()
        return chain
        
