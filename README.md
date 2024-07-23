# LLM Project

# Configuration Guide

```
# rag.api.py
_config = {
    "transformation": { # optional
        "model": "gpt-4o-mini", # model id
        "enable": {
            "translation": True,
            "rewriting": False,
            "expansion": False,
            "hyde": False,
        },
    },
    "retrieval": { # mandatory
        "retriever": ["pinecone"], # retriever alias or list of retrievers with weights
        # "retriever": ["pinecone", "kendra"]
        # "weights": [0.6, 0.4]
        "embedding": "amazon.titan-embed-text-v1", # model id. may be optional
        "top_k": 5,
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