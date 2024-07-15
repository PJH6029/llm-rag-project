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
        write_source_docs(rag_api.recent_base_docs, title="Base Documents")
        write_source_docs(rag_api.recent_additional_docs, title="Additional Documents")

    st.session_state.messages.append(
        {
            "role": "AI",
            "content": response
        }
    )

        # TODO visualize pipline
