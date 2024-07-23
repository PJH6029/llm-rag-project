from wasabi import msg
from typing import Generator, Optional

from langchain_core.output_parsers import StrOutputParser

from rag.model import llm, prompt
from rag.managers.base import BasePipelineManager
from rag.type import Chunk, ChatLog
from rag import util

class GeneratorManager(BasePipelineManager):
    def __init__(self) -> None:
        super().__init__()
        self.generator_name = None
    
    def set_config(self, config: dict):
        self.generator_name = config.get("model")
        
        msg.info(f"Setting GENERATOR to {self.generator_name}")

    def generate_stream(
        self, query: str, history: str="", context: str=""
    ) -> Generator[str, None, None]:
        if self.generator_name is None:
            msg.warn("Generator not set. Skipping generation.")
            return
        
        generator = llm.get_model(self.generator_name)
        if generator is None:
            msg.warn(f"Generator {self.generator_name} not found. Skipping generation.")
            return
        
        chain = prompt.generation_prompt | generator | StrOutputParser()
        for r in chain.stream({"query": query, "context": context, "history": history}):
            yield r
    
    def generate(
        self, query: str, history: str="", context: str=""
    ) -> str:
        if self.generator_name is None:
            msg.warn("Generator not set. Skipping generation.")
            return ""
        
        generator = llm.get_model(self.generator_name, temperature=0.0)
        if generator is None:
            msg.warn(f"Generator {self.generator_name} not found. Skipping generation.")
            return ""
        
        chain = prompt.generation_prompt | generator | StrOutputParser()
        return chain.invoke({"query": query, "context": context, "history": history})
