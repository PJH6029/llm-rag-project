import os
from wasabi import msg

from rag.generate.GPT4Generator import GPT4Generator
from rag.prompts import generation_prompt

class GPT3Generator(GPT4Generator):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.context_window = 15000
