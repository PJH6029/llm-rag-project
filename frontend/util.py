import streamlit as st
from wasabi import msg

from langchain_core.messages import AIMessage, HumanMessage

from rag.type import Chunk, CombinedChunks, TransformationResult
from rag.util import combine_chunks

def session_init(session_state):
    if "messages" not in session_state:
        session_state.messages = []
    if "translated_messages" not in session_state:
        session_state.translated_messages = []

def display_chat_history(session_state):
    for message in session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def write_queries(queries: TransformationResult):
    for i, k in enumerate(queries):
        st.markdown(f"- query {i+1}:")
        st.markdown(f"```\n{queries[k]}\n```")

def write_chunks(chunks: list[Chunk]):
    for i, chunk in enumerate(chunks):
        st.markdown(f"- chunk {i+1}:")
        st.markdown(f"```\n{chunk.text[:70]}...\n```")

def _write_source_docs_without_hierarchy(chunks: list[Chunk]):
    combined_chunks = combine_chunks(chunks, attach_url=True)
    
    st.divider()
    st.markdown(f"# Total {len(chunks)} chunks")
    write_combined_chunks(combined_chunks)

def write_source_docs(chunks: list[Chunk]):
    if not all([chunk.doc_meta.get("category") for chunk in chunks]):
        msg.warn("Some chunks do not have category information, cannot display hierarchy")
        _write_source_docs_without_hierarchy(chunks)
        return
    
    combined_chunks = combine_chunks(chunks, attach_url=True)
    
    combined_base_chunks = [chunk for chunk in combined_chunks if chunk.doc_meta.get("category") == "base"]
    combined_additional_chunks = [chunk for chunk in combined_chunks if chunk.doc_meta.get("category") == "additional"]
    
    # base
    st.divider()
    st.markdown(f"# Base Documents({sum([len(combined_chunk.chunks) for combined_chunk in combined_base_chunks])} chunks)")
    write_combined_chunks(combined_base_chunks)
    
    # additional
    st.divider()
    st.markdown(f"# Additional Documents({sum([len(combined_chunk.chunks) for combined_chunk in combined_additional_chunks])} chunks)")
    write_combined_chunks(combined_additional_chunks)

def write_combined_chunks(combined_chunks: list[CombinedChunks]) -> None:
    for i, combined_chunk in enumerate(combined_chunks):
        st.markdown(f"## Document: {combined_chunk.doc_meta.get('doc_name', '')}")
        if combined_chunk.doc_meta.get("base_doc_id"):
            st.write(f"Based on: {combined_chunk.doc_meta.get('base_doc_id')}")
        st.write(f"Average Score: {combined_chunk.doc_mean_score:.2f}")
        if combined_chunk.link:
            st.write(f"URL: [link]({combined_chunk.link})")
        for chunk in (combined_chunk.chunks):
            st.markdown(f"### Chunk ({chunk.chunk_id})")
            st.write(f"#### Score: {chunk.score:.2f}")
            st.write(f"#### Content")
            if chunk.chunk_meta.get("page"):
                try:
                    st.write(f"Page: {int(chunk.chunk_meta['page'])}")
                except ValueError:
                    msg.warn(f"Page number is not an integer: {chunk.chunk_meta['page']}")
                    
            st.write(chunk.text)
        
        if i < len(combined_chunks) - 1:
            st.divider()