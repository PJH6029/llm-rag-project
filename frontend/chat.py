import streamlit as st
from wasabi import msg
import time

from frontend.util import *
from rag.api import api as rag_api

def run():
    session_init(st.session_state)
    display_chat_history(st.session_state)

    if user_input := st.chat_input("Enter a message (줄임말은 정확도를 떨어뜨릴 수 있습니다)..."):
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            
            rag_generator = rag_api.query_stream(query=user_input, history=st.session_state.translated_messages[:])
            
            with st.status("Understaning question...") as status:
                start = time.time()
                transformed_queries = next(rag_generator).get("transformation")
                if transformed_queries:
                    write_queries(transformed_queries)
                else:
                    msg.warn("No transformation found. Check the query.")
                end = time.time()
                status.update(label=f"Understanding question... ({end-start:.2f} seconds)")
                # status.update(label=f"Transformed into {len(transformed_queries)} queries in {end-start:.2f} seconds.")
            
            with st.status("Looking for relevant contents...") as status:
                start = time.time()
                chunks = next(rag_generator).get("retrieval")
                if chunks:
                    write_chunks(chunks)
                else:
                    msg.warn("No data found. Check the retrieval result.")
                end = time.time()
                status.update(label=f"Looking for relevant contents... ({end-start:.2f} seconds)")
                # status.update(label=f"{len(chunks)} chunks retrieved in {end-start:.2f} seconds.")
            
        
            response = st.write_stream(
                (response.get("generation") for response in rag_generator)
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
