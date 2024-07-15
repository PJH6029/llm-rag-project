from dotenv import load_dotenv
import os
from wasabi import msg

from rag.managers import RAGManager
from rag import utils
from rag.types import *

manager: RAGManager = None

recent_base_docs = None
recent_additional_docs = None

def init(config: dict=None):
    load_dotenv()
    global manager
    manager = RAGManager()
    # config = utils.load_config()
    _config = {
        "model": {
            "Reader": {"selected": ""},
            "Chunker": {"selected": ""},
            "Embedder": {"selected": ""},
            "Retriever": {"selected": "knowledge-base"},
            "Generator": {"selected": "gpt4"},
            "Revisor": {"selected": "gpt"},
        },
        "pipeline": {
            "revise_query": True,
            "hyde": True,
        }
    } if config is None else config
    manager.set_config(_config)

init()

def retrieve_config():
    pass

def reset():
    pass

def query():
    pass

def query_stream(query: str, history: list[dict]=None):
    msg.info(f"Querying with: {query} and {len(history)} history...")
    base_chunks, additional_chunks, context = manager.retrieve_chunks([query], history=history)
    for response in manager.generate_stream_answer([query], context, history):
        yield response
        # TODO fact verification
    
    global recent_base_docs, recent_additional_docs
    recent_base_docs = manager.combine_chunks(base_chunks, doc_type="base", attach_url=True)
    recent_additional_docs = manager.combine_chunks(additional_chunks, doc_type="additional", attach_url=True)
    msg.good(f"Query completed")

def index_document(file: FileData):
    pass 
    # TODO metadata

def index_all_documents(files: list[FileData]):
    pass

def retrieve_document(queries: list[str]):
    pass

def retrieve_all_documents(queries: list[str]):
    pass

def get_document_by_id(doc_id: str):
    pass

def get_documents_by_ids(doc_ids: list[str] | str="all"):
    pass