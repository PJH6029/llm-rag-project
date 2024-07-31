from wasabi import msg

from langchain_core.output_parsers import StrOutputParser

from rag.component import llm, prompt
from rag.managers.base import BasePipelineManager
from rag.type import *

class TransformerManager(BasePipelineManager):       
    def __init__(self) -> None:
        super().__init__()
        self.transformer_name = None
        self.enable = {}
     
    def set_config(self, config: dict):
        self.transformer_name = config.get("model")
        self.enable = config.get("enable", {})
        
        msg.info(f"Setting TRANSFORMER to {self.transformer_name}")
        
    def transform(self, sentence: str, history: list[ChatLog]=None) -> list[str]:
        if self.transformer_name is None:
            msg.warn("Transformer not set. Skipping transformation.")
            return [sentence]

        history = history or []
        sentences = []
        
        if self.enable.get("translation", False):
            sentence = self.translate(sentence)
            sentences.append(sentence)
        else:
            sentences.append(sentence)
        
        if self.enable.get("rewriting", False):
            sentence = self.rewrite(sentence, history)
            sentences.append(sentence)
        if self.enable.get("expansion", False):
            expanded_sentences = self.expand(sentence, history)
            sentences.extend(expanded_sentences)
        if self.enable.get("hyde", False):
            sentence = self.hyde(sentence, history)
            sentences.append(sentence)

        return list(dict.fromkeys(sentences)) # remove duplicates, preserving order
    
    def translate(self, sentence: str) -> str:
        transformer = llm.get_model(self.transformer_name)
        if transformer is None:
            return sentence
        
        chain = prompt.translation_prompt | transformer | StrOutputParser()
        return chain.invoke({"sentence": sentence})
    
    def rewrite(self, sentence: str, history: list[ChatLog]) -> str:
        transformer = llm.get_model(self.transformer_name)
        if transformer is None:
            return sentence
        
        chain = prompt.rewrite_prompt | transformer | StrOutputParser()
        return chain.invoke({"query": sentence, "history": history})
    
    def expand(self, sentence: str, history: list[ChatLog]) -> list[str]:
        transformer = llm.get_model(self.transformer_name)
        if transformer is None:
            return [sentence]
        
        chain = prompt.expansion_prompt | transformer | StrOutputParser()
        sentences = chain.invoke({"query": sentence, "history": history})
        try:
            return sentences.split(",")
        except:
            return [sentence]
    
    def hyde(self, sentence: str, history: list[ChatLog]) -> str:
        transformer = llm.get_model(self.transformer_name)
        if transformer is None:
            return sentence
        
        chain = prompt.hyde_prompt | transformer | StrOutputParser()
        return chain.invoke({"query": sentence, "history": history})