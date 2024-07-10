from dotenv import load_dotenv
import os
from wasabi import msg

from rag.managers import RAGManager
from rag import utils
from rag.types import *

manager: RAGManager = None

recent_base_docs = None
recent_additional_docs = None

def _init():
    load_dotenv()
    global manager
    manager = RAGManager()
    config = utils.load_config()
    manager.init(config)

_init()

def retrieve_config():
    pass

def reset():
    pass

def query():
    pass

def query_stream(query: str, history: list[dict]=None):
    msg.info(f"Querying with: {query} and {len(history)} history...")
    base_chunks, additional_chunks, context = manager.retrieve_chunks([query], history=history, revise_query=True)
    for response in manager.generate_stream_answer([query], context, history):
        yield response
        # TODO fact verification
    
    global recent_base_docs, recent_additional_docs
    recent_base_docs = manager.combine_chunks(base_chunks)
    recent_additional_docs = manager.combine_chunks(additional_chunks)
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