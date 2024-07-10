import os
from dotenv import load_dotenv
from wasabi import msg

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

from rag.interfaces import Revisor
from rag.prompts import revision_prompt
from rag import utils

load_dotenv()

class GPTRevisor(Revisor):
    def __init__(self) -> None:
        super().__init__()
        self.prompt = revision_prompt
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.context_window = 15000
    
    def revise(self, queries: list[str], history: list[dict], revise_prompt: ChatPromptTemplate=None) -> list[str]:
        try:
            from langchain_openai import ChatOpenAI

            openai = ChatOpenAI(model=self.model_name) # TODO temperature
            msg.info(f"OpenAI Model {self.model_name} initialized. Revising...")

            history_str = utils.history_to_str(history)

            prompt = revise_prompt or self.prompt

            chain = prompt | openai | StrOutputParser()
            return chain.invoke({
                "query": queries[0], # TODO
                "history": history_str
            })
        except Exception as e:
            raise e