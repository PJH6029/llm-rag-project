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
        
        self.user_lang = config.get("lang", {}).get("user", "Korean")
        self.source_lang = config.get("lang", {}).get("source", "English")
        
        msg.info(f"Setting TRANSFORMER to {self.transformer_name}")
 
    def translate(self, sentence: str) -> str:
        transformer = llm.get_model(self.transformer_name)
        if transformer is None:
            return sentence
        
        chain = prompt.translation_prompt.partial(user_lang=self.user_lang, source_lang=self.source_lang) | transformer | StrOutputParser()
        return chain.invoke({"sentence": sentence})
    
    def transform(self, sentence: str, history: list[ChatLog]=None) -> TransformationResult:
        if self.transformer_name is None:
            msg.warn("Transformer not set. Skipping transformation.")
            return [sentence]

        history = history or []
        sentences: TransformationResult = {}
        
        chains = {}
        
        # if translation is disabled even though the user language is different from the source language,
        # the entire queries will be in the user language
        if self.user_lang != self.source_lang and self.enable.get("translation", False):
            sentence = self.translate(sentence)
            sentences["translation"] = sentence
            query_lang = self.source_lang
        else:
            sentences["translation"] = sentence
            query_lang = self.user_lang
        
        for key in ["rewriting", "expansion", "hyde"]:
            if not self.enable.get(key, False):
                continue
            
            chain = self.build_chain(key, lang=query_lang, temperature=0.9)
            
            if chain is not None:
                if key == "expansion":
                    chain = chain | RunnableLambda(split_lambda)
                chains[key] = chain
        
        parallel_chain = RunnableParallel(**chains)
        transformed_sentences = parallel_chain.invoke({"query": sentence, "history": history})
        for key, _sentence in transformed_sentences.items():
            sentences[key] = _sentence

        return sentences

    def build_chain(self, key: str, *, lang: str="English", **model_kwargs: dict) -> Optional[Runnable]:
        prompts = {
            "rewriting": prompt.rewrite_prompt.partial(lang=lang),
            "expansion": prompt.expansion_prompt.partial(lang=lang),
            "hyde": prompt.hyde_prompt.partial(lang=lang),
        }
        transformer = llm.get_model(self.transformer_name, **model_kwargs)
        if transformer is None:
            return None
        
        chain = prompts[key] | transformer | StrOutputParser()
        
        return chain
    