import streamlit as st

from frontend.utils import *
from rag.api import api as rag_api

session_init(st.session_state)
display_chat_history(st.session_state)

if user_input := st.chat_input("Enter a message..."):
    st.session_state.messages.append(
        {
            "role": "Human",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        response = st.write_stream(
            # to_sync_generator(rag_api.query_stream(query=user_input, history=st.session_state.messages[:]))
            rag_api.query_stream(query=user_input, history=st.session_state.messages[:])
        )
    
    with st.sidebar:
        st.markdown("# Source Documents")
        st.divider()
        for doc_id in rag_api.recent_chunks:
            doc = rag_api.recent_chunks[doc_id]
            st.markdown(f'## {doc["doc_name"]}')
            st.write(f"Docs score: {doc['score']}")
            for idx, chunk_id in enumerate(doc["chunks"]):
                st.markdown(f"### Chunk {idx}")
                st.write(f"Content")
                st.write(doc["chunks"][chunk_id].text[:max(len(doc["chunks"][chunk_id].text) // 3, 300)])
            st.divider()

    st.session_state.messages.append(
        {
            "role": "AI",
            "content": response
        }
    )

        # TODO visualize pipline
