from wasabi import msg
from typing import Optional

from rag.type import AnyLanguageModel

model_providers = {
    "openai": [
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
    ],
    "anthropic": [
        "claude", # TODO
    ]
}

def get_provider(model_name: str) -> Optional[str]:
    for provider, models in model_providers.items():
        if model_name in models:
            return provider
    return None

def get_model(model_name: str, **kwargs) -> Optional[AnyLanguageModel]:
    try:
        provider = get_provider(model_name)
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, **kwargs)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_name, **kwargs)
        else:
            msg.fail(f"Provider {provider} not supported.")
            return None
    except Exception as e:
        msg.fail(f"Error loading model {model_name} from provider {provider}. Make sure you have installed the required packages.")
        return None