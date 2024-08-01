# LLM Project

# Configuration Guide

```python
# rag.api.py
_config = {
    "global": {
        "context-hierarchy": True, # used in selecting retriever and generation prompts
    },
    "ingestion": { # optional
        "ingestor": "pinecone-multivector",
        "embeddings": "text-embedding-3-small",
        "namespace": "parent-upstage-overlap-backup",
        "sub-namespace": "child-upstage-overlap-backup",
    },
    "transformation": { # optional
        "model": "gpt-4o-mini",
        "enable": {
            "translation": True,
            "rewriting": True,
            "expansion": False,
            "hyde": True,
        },
    },
    "retrieval": { # mandatory
        # "retriever": ["pinecone-multivector", "kendra"],
        "retriever": ["pinecone-multivector"],
        # "retriever": ["kendra"],
        # "weights": [0.5, 0.5],
        "namespace": "parent-upstage-overlap-backup",
        "sub-namespace": "child-upstage-overlap-backup",
        
        "embeddings": "text-embedding-3-small", # may be optional
        "top_k": 3, # for multi-vector retriever, context size is usually big. Use small top_k
        "post_retrieval": {
            "rerank": True,
            # TODO
        }
    },
    "generation": { # mandatory
        "model": "gpt-4o",
    },
    "fact_verification": { # optional
        "model": "gpt-4o-mini",
        "enable": False
    }
}
```