from typing import Generator, Any
from wasabi import msg

from langchain_community.callbacks import get_openai_callback
from langchain.globals import set_debug

from rag.rag_manager import RAGManager
from rag import util
from rag.type import *

recent_chunks = None
recent_translated_query = None
rag_manager = None

def init(config: dict=None):
    # set_debug(True)
    
    util.load_secrets()
    global rag_manager
    rag_manager = RAGManager()
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
            "retriever": ["kendra"],
            # "weights": [0.5, 0.5],
            # "embedding": "amazon.titan-embed-text-v1", # may be optional
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
    } if config is None else config
    
    rag_manager.set_config(_config)

init()

def _setup_generation_params(query: str, history: list[ChatLog]) -> tuple[str, list[ChatLog], list[Chunk]]:
    queries = rag_manager.transform_query(query, history)
    translated_query = queries[0] # first query is the translated query
    chunks = rag_manager.retrieve(queries)
    
    global recent_chunks, recent_translated_query
    recent_chunks = chunks
    recent_translated_query = translated_query
    
    return translated_query, history, chunks
    

def query(query: str, history: list[ChatLog]=None) -> GenerationResult:
    history = history or []
    with get_openai_callback() as cb:
        translated_query, history, chunks = _setup_generation_params(query, history)

        generation_response = rag_manager.generate(translated_query, history, chunks)
        verification_response = rag_manager.verify_fact(generation_response, chunks)
        
        print(cb)
    return {"generation": generation_response, "fact_verification": verification_response}
    

def query_stream(query: str, history: list[ChatLog]=None) -> Generator[GenerationResult, None, None]:
    history = history or []
    with get_openai_callback() as cb:
        translated_query, history, chunks = _setup_generation_params(query, history)

        generation_response = ""
        for response in rag_manager.generate_stream(translated_query, history, chunks):
            yield {"generation": response}
            generation_response += response
        
        for response in rag_manager.verify_fact_stream(generation_response, chunks):
            yield {"fact_verification": response}
        
        print(cb)
    
def upload_data(data: Any):
    # 1. upload to s3
    # 2. ingest
    pass    

def ingest_data(data: Any):
    # 1. download to local from s3 with metadata
    # 2. split
    # 3. embed
    # 4. upload to vectorstore
    pass

# TODO ingesting logic may be different between vectorstores. Need to abstract this.