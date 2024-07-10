import os
from dotenv import load_dotenv
from wasabi import msg

from langchain_core.output_parsers import StrOutputParser

from rag.interfaces import Generator
from rag.prompts import generation_prompt_v2
from rag import utils

load_dotenv()

class GPT4Generator(Generator):
    def __init__(self) -> None:
        super().__init__()
        self.prompt = generation_prompt_v2
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        self.context_window = 20000


    def generate(self, queries: list[str], context: str, history: dict=None) -> list[str]:
        pass

    def generate_stream(self, queries: list[str], context: dict[str, str], history: list[dict]=None):
        try:
            from langchain_openai import ChatOpenAI
            
            openai = ChatOpenAI(model=self.model_name, temperature=0.0, stream_usage=True)
            msg.info(f"OpenAI Model {self.model_name} initialized. Generating stream for queries: {queries}...")

            history_str = utils.history_to_str(history)

            chain = self.prompt | openai | StrOutputParser()
            return chain.stream({
                "query": queries[0], # TODO
                "base_context": context["base"],
                "additional_context": context["additional"],
                "history": history_str
            })
        except Exception as e:
            raise e