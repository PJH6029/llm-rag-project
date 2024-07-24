# LLM Project

# Configuration Guide

```
# rag.api.py
_config = {
    "global": {
        "context-hierarchy": False, # used in selecting retriever and generation prompts
    },
    "transformation": { # optional
        "model": "gpt-4o-mini",
        "enable": {
            "translation": True,
            "rewriting": False,
            "expansion": False,
            "hyde": False,
        },
    },
    "retrieval": { # mandatory
        "retriever": ["pinecone", "knowledge-base-pinecone"],
        # "weights": [0.5, 0.5],
        "embedding": "amazon.titan-embed-text-v1", # may be optional
        "top_k": 7,
        "post_retrieval": {
            "rerank": True,
            # TODO
        }
    },
    "generation": { # mandatory
        "model": "gpt-4o-mini",
    },
    "fact_verification": { # optional
        "model": "gpt-4o-mini",
        "enable": False
    },
}
```