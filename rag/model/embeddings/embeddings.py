from wasabi import msg
from typing import Optional
import os

from langchain_core.embeddings import Embeddings

model_providers = {
    "openai": [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002"
    ],
    "bedrock": [
        "amazon.titan-embed-text-v1",
        "amazon.titan-text-express-v1",
    ]
}

def get_provider(model_name: str) -> Optional[str]:
    for provider, models in model_providers.items():
        if model_name in models:
            return provider
    return None

def get_model(model_name: str, **kwargs) -> Optional[Embeddings]:
    try:
        provider = get_provider(model_name)
        if provider is None:
            msg.fail(f"Cannot find provider for model {model_name}.")
            return None

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model_name, **kwargs)
        elif provider == "bedrock":
            from langchain_community.embeddings import BedrockEmbeddings
            return BedrockEmbeddings(model_id=model_name, region_name=os.environ["AWS_REGION"], **kwargs)
        else:
            msg.fail(f"Provider {provider} not supported.")
            return None
    except Exception as e:
        print(e)
        msg.fail(f"Error loading model {model_name} from provider {provider}. Make sure you have installed the required packages.")
        return None