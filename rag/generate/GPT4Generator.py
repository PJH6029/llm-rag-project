import os
from dotenv import load_dotenv
from wasabi import msg

from langchain_core.output_parsers import StrOutputParser

from rag.interfaces import Generator
from rag.prompts import generation_prompt

load_dotenv()

class GPT4Generator(Generator):
    def __init__(self) -> None:
        super().__init__()
        self.prompt = generation_prompt
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        self.context_window = 20000

    def generate(self, queries: list[str], context: str, history: dict=None) -> list[str]:
        pass

    def generate_stream(self, queries: list[str], context: str, history_str: str="No history"):
        try:
            from langchain_openai import ChatOpenAI
            
            openai = ChatOpenAI(model=self.model_name, temperature=0.0, stream_usage=True)
            msg.info(f"OpenAI Model {self.model_name} initialized. Generating stream for queries: {queries}...")

            chain = self.prompt | openai | StrOutputParser()
            return chain.stream({
                "query": queries[0], # TODO
                "context": context,
                "history": history_str
            })
        except Exception as e:
            raise e