import streamlit as st

from frontend.util import *
from rag.api import api as rag_api

def run():
    session_init(st.session_state)
    display_chat_history(st.session_state)

    if user_input := st.chat_input("Enter a message..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            response = st.write_stream(
                (r["generation"] for r in rag_api.query_stream(query=user_input, history=st.session_state.translated_messages[:]))
            )
        
        with st.sidebar:
            st.markdown("# Source Documents")
            write_source_docs(rag_api.recent_chunks)

        st.session_state.messages.append(
            {
                "role": "Human",
                "content": user_input
            }
        )
        st.session_state.translated_messages.append(
            {
                "role": "Human",
                "content": rag_api.recent_translated_query
            }
        )


        st.session_state.messages.append(
            {
                "role": "AI",
                "content": response
            }
        )
        st.session_state.translated_messages.append(
            {
                "role": "AI",
                "content": response
            }
        )

        # TODO visualize pipline