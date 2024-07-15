from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from rag.types import CombinedChunks
# from typing import AsyncGenerator
# import asyncio

def session_init(session_state):
    if "messages" not in session_state:
        session_state.messages = []

def display_chat_history(session_state):
    for message in session_state.messages:
        role = "AI" if message["role"] == "AI" else "Human"
        with st.chat_message(role):
            st.markdown(message["content"])

def write_source_docs(docs: dict[str, CombinedChunks], title=""):
    st.divider()
    st.markdown(f"## {title}")
    st.write(f"Total {sum([docs[doc_id].num_chunks for doc_id in docs])} chunks")
    for doc_id in docs:
        doc = docs[doc_id]
        st.markdown(f'## {doc.doc_name}')
        st.write(f"Total doc score: {doc.score}")
        if doc.url:
            st.write(f"URL: {doc.url}")
        for idx, chunk_id in enumerate(doc.chunks):
            st.markdown(f"### Chunk {idx}")
            st.write(f"Content")
            # st.write(doc["chunks"][chunk_id].text[:max(len(doc["chunks"][chunk_id].text) // 3, 300)])
            st.write(doc.chunks[chunk_id].text)
            st.write(f"Excerpt Page: {doc.chunks[chunk_id].chunk_meta['excerpt_page_number']}")
            if doc.chunks[chunk_id].chunk_meta.get("base_doc_id"):
                st.write(f"Base Document ID: {doc.chunks[chunk_id].chunk_meta['base_doc_id']}")
        st.divider()

# def to_sync_generator(async_gen: AsyncGenerator):
#     while True:
#         try:
#             yield asyncio.run(anext(async_gen))
#         except StopAsyncIteration:
#             break