import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage

from rag.type import Chunk
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

def write_source_docs(chunks: list[Chunk]):
    combined_chunks = combine_chunks(chunks, attach_url=True)
    
    combined_base_chunks = [chunk for chunk in combined_chunks if chunk.doc_meta.get("category") == "base"]
    combined_additional_chunks = [chunk for chunk in combined_chunks if chunk.doc_meta.get("category") == "additional"]
    
    # base
    st.divider()
    st.markdown(f"## Base Documents({sum([len(combined_chunk.chunks) for combined_chunk in combined_base_chunks])} chunks)")
    for combined_chunk in combined_base_chunks:
        st.markdown(f"### Document: {combined_chunk.doc_meta.get('doc_name', '')}")        
        st.write(f"Average Score: {combined_chunk.doc_mean_score:.2f}")
        if combined_chunk.link:
            st.write(f"URL: {combined_chunk.link}")
        for chunk in (combined_chunk.chunks):
            st.markdown(f"#### Chunk ({chunk.chunk_id})")
            st.write(f"Score: {chunk.score:.2f}")
            st.write(f"Content")
            st.write(chunk.text)
            
            if chunk.chunk_meta.get("page"):
                st.write(f"Page: {chunk.chunk_meta['page']}")
    
    # additional
    st.divider()
    st.markdown(f"## Additional Documents({sum([len(combined_chunk.chunks) for combined_chunk in combined_additional_chunks])} chunks)")
    for combined_chunk in combined_additional_chunks:
        st.markdown(f"### Document: {combined_chunk.doc_meta.get('doc_name', '')}")
        st.write(f"Based on: {combined_chunk.doc_meta.get('base_doc_id', '')}")
        st.write(f"Average Score: {combined_chunk.doc_mean_score:.2f}")
        if combined_chunk.link:
            st.write(f"URL: {combined_chunk.link}")
        for chunk in (combined_chunk.chunks):
            st.markdown(f"#### Chunk ({chunk.chunk_id})")
            st.write(f"Score: {chunk.score:.2f}")
            st.write(f"Content")
            st.write(chunk.text)
            
            if chunk.chunk_meta.get("page"):
                st.write(f"Page: {chunk.chunk_meta['page']}")
