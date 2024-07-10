import tiktoken
from rag.types import *

def load_config():
    return {}

def truncate_history(history: list[dict], max_tokens: int) -> dict:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    sum_tokens = 0
    truncated_history = []

    # Start with newest history
    for item in reversed(history):
        tokens = encoding.encode(item["content"], disallowed_special=())

        if sum_tokens + len(tokens) > max_tokens:
            remaining_tokens = max_tokens - sum_tokens
            truncated_tokens = tokens[:remaining_tokens]
            truncated_content = encoding.decode(truncated_tokens)

            truncated_item = {
                "role": item["role"],
                "content": truncated_content,
            }
            truncated_history.append(truncated_item)
            break
        else:
            sum_tokens += len(tokens)
            truncated_history.append(item)
    return list(reversed(truncated_history))

def history_to_str(history: list[dict]) -> str:
    return "\n".join([f"{item['role'].upper()}: {item['content']}" for item in history])
