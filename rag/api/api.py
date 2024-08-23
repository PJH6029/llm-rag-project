from typing import Generator, Any
from wasabi import msg

from langchain_community.callbacks import get_openai_callback
from langchain.globals import set_debug

from rag.rag_manager import RAGManager
from rag import util
from rag.type import *
from rag.config import *
from rag.component.loader import *

recent_chunks = None
recent_translated_query = None
rag_manager = None

def init():    
    util.load_secrets()
    global rag_manager
    rag_manager = RAGManager()
    
    config_dict = util.load_config()
    if config_dict is None:
        msg.warn("No config provided. Using default config.")
        _config = RAGConfig()
    else:
        try:
            _config = RAGConfig(**config_dict)
        except Exception as e:
            msg.warn(f"Invalid config provided. Using default config. {e}")
            _config = RAGConfig()
    rag_manager.set_config(_config)

init()

def get_config() -> Optional[RAGConfig]:
    return rag_manager.config

def query(query: str, history: list[ChatLog]=None) -> GenerationResult:
    history = history or []
    with get_openai_callback() as cb:
        queries = rag_manager.transform_query(query, history)
        translated_query = queries["translation"]
        chunks = rag_manager.retrieve(queries)
        
        global recent_chunks, recent_translated_query
        recent_chunks = chunks
        recent_translated_query = translated_query

        generation_response = rag_manager.generate(translated_query, history, chunks)
        verification_response = rag_manager.verify_fact(generation_response, chunks)
        
        print(cb)
    return util.remove_falsy({"transformation": queries, "retrieval": chunks, "generation": generation_response, "fact_verification": verification_response})
    

def query_stream(
    query: str, 
    history: list[ChatLog]=None,
    categories: list[str]=None,
) -> Generator[GenerationResult, None, None]:
    history = history or []
    with get_openai_callback() as cb:
        queries = rag_manager.transform_query(query, history)
        yield {"transformation": queries}
        
        translated_query = queries["translation"]
        chunks = rag_manager.retrieve(queries, categories)
        yield {"retrieval": chunks}
        
        global recent_chunks, recent_translated_query
        recent_chunks = chunks
        recent_translated_query = translated_query

        generation_response = ""
        for response in rag_manager.generate_stream(translated_query, history, chunks):
            yield {"generation": response}
            generation_response += response
        
        verification = rag_manager.verify_fact(generation_response, chunks)
        if verification is not None:
            yield {"fact_verification": verification}
        
        print(cb)
        
def fake_query_stream(
    query: str, 
    history: list[ChatLog]=None,
    categories: list[str]=None,
) -> Generator[GenerationResult, None, None]:
    global recent_chunks, recent_translated_query
    recent_chunks = [Chunk(text="chunk 1", doc_id="doc#1", chunk_id="chunk#1"), Chunk(text="chunk 2", doc_id="doc#2", chunk_id="chunk#2")]
    recent_translated_query = "translated query"
    
    yield {"transformation": {"translation": "translated query"}}
    yield {"retrieval": [Chunk(text="chunk 1", doc_id="doc#1", chunk_id="chunk#1"), Chunk(text="chunk 2", doc_id="doc#2", chunk_id="chunk#2")]}
    yield {"generation": "generated response"}
    yield {"fact_verification": VerificationResult(**{"verification": True, "reasoning": "reasoning"})}
    
def upload_data(file_path: str, object_location: str, metadata: Optional[dict] = None) -> bool:
    return rag_manager.upload_data(file_path, object_location, metadata)

def ingest_data(loader: Union[BaseRAGLoader, BaseLoader], batch_size: int = 20) -> int:
    return rag_manager.ingest(loader=loader, batch_size=batch_size)

async def aingest_data(s3_url: str) -> int:
    return await rag_manager.aingest(s3_url)

def get_categories() -> list[str]:
    return rag_manager.get_categories()
