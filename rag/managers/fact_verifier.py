from typing import Optional, Generator
from wasabi import msg

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from rag.managers.base import BasePipelineManager
from rag.type import VerificationResult
from rag import util
from rag.component import llm, prompt

class FactVerifierManager(BasePipelineManager):
    def __init__(self) -> None:
        super().__init__()
        self.verifier_name = None
        self.enable = False
        
    def set_config(self, config: dict):
        self.verifier_name = config.get("model")
        self.enable = config.get("enable", False)
        self.user_lang = config.get("lang", {}).get("user", "Korean")
        
        self.prompt = prompt.verification_prompt.partial(lang=self.user_lang)
        
        msg.info(f"Setting FACT_VERIFIER to {self.verifier_name}")

    # TODO: Implement verification stream
    # def verify_stream(self, response: str, context: str) -> Generator[VerificationResult, None, None]:
    #     if not self.enable:
    #         msg.warn("Fact verifier not enabled. Skipping verification.")
    #         return
            
    #     if self.verifier_name is None:
    #         msg.warn("Fact verifier not set. Skipping verification.")
    #         return
        
    #     verifier = llm.get_model(self.verifier_name)
    #     if verifier is None:
    #         msg.warn(f"Verifier {self.verifier_name} not found. Skipping verification.")
    #         return
                
    #     chain = self.prompt | verifier | StrOutputParser()
    #     for r in chain.stream({"response": response, "context": context}):
    #         yield r
        
    def verify(self, response: str, context: str) -> Optional[VerificationResult]:
        if not self.enable:
            msg.warn("Fact verifier not enabled. Skipping verification.")
            return None
        
        if self.verifier_name is None:
            msg.warn("Fact verifier not set. Skipping verification.")
            return None
        
        verifier = llm.get_model(self.verifier_name)
        if verifier is None:
            msg.warn(f"Verifier {self.verifier_name} not found. Skipping verification.")
            return None
                
        chain = self.prompt | verifier | JsonOutputParser(pydantic_object=VerificationResult)
        response = chain.invoke({"response": response, "context": context})
        return VerificationResult.model_validate(response)
