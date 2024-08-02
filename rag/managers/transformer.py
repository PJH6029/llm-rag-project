from typing import Optional
from wasabi import msg

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel, RunnableLambda

from rag.component import llm, prompt
from rag.managers.base import BasePipelineManager
from rag.type import *

def split_lambda(x: str) -> list[str]:
    return list(filter(lambda x: bool(x), x.split("\n")))

class TransformerManager(BasePipelineManager):       
    def __init__(self) -> None:
        super().__init__()
        self.transformer_name = None
        self.enable = {}
     
    def set_config(self, config: dict):
        self.transformer_name = config.get("model")
        self.enable = config.get("enable", {})
        
        msg.info(f"Setting TRANSFORMER to {self.transformer_name}")
        
    def transform_deprecated(self, sentence: str, history: list[ChatLog]=None) -> list[str]:
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
    
    def transform(self, sentence: str, history: list[ChatLog]=None) -> TransformationResult:
        if self.transformer_name is None:
            msg.warn("Transformer not set. Skipping transformation.")
            return [sentence]

        history = history or []
        sentences: TransformationResult = {}
        
        chains = {}
        
        if self.enable.get("translation", False):
            sentence = self.translate(sentence)
            sentences["translation"] = sentence
        else:
            sentences["translation"] = sentence
        
        for key in ["rewriting", "expansion", "hyde"]:
            if not self.enable.get(key, False):
                continue
            
            chain = self.build_chain(key, temperature=0.9)
            
            if chain is not None:
                if key == "expansion":
                    chain = chain | RunnableLambda(split_lambda)
                chains[key] = chain
        
        parallel_chain = RunnableParallel(**chains)
        transformed_sentences = parallel_chain.invoke({"query": sentence, "history": history})
        for key, _sentence in transformed_sentences.items():
            sentences[key] = _sentence

        return sentences

    def build_chain(self, key: str, **model_kwargs: dict) -> Optional[Runnable]:
        prompts = {
            "rewriting": prompt.rewrite_prompt,
            "expansion": prompt.expansion_prompt,
            "hyde": prompt.hyde_prompt
        }
        transformer = llm.get_model(self.transformer_name, **model_kwargs)
        if transformer is None:
            return None
        
        chain = prompts[key] | transformer | StrOutputParser()
        
        return chain
    