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
        # st.divider()
        # st.markdown("## Base Documents")
        # for doc_id in rag_api.recent_base_chunks:
        #     doc = rag_api.recent_base_chunks[doc_id]
        #     st.markdown(f'## {doc["doc_name"]}')
        #     st.write(f"Docs score: {doc['score']}")
        #     for idx, chunk_id in enumerate(doc["chunks"]):
        #         st.markdown(f"### Chunk {idx}")
        #         st.write(f"Content")
        #         # st.write(doc["chunks"][chunk_id].text[:max(len(doc["chunks"][chunk_id].text) // 3, 300)])
        #         st.write(doc["chunks"][chunk_id].text)
        #         st.write(f"Excerpt Page: {doc['chunks'][chunk_id].chunk_meta['excerpt_page_number']}")
        #         if doc["chunks"][chunk_id].chunk_meta.get("base_doc_id"):
        #             st.write(f"Base Document ID: {doc['chunks'][chunk_id].chunk_meta['base_doc_id']}")
        #     st.divider()
        write_source_docs(rag_api.recent_base_docs, title="Base Documents")
        
        # st.divider()
        # st.markdown("## Additional Documents")
        # for doc_id in rag_api.recent_additional_chunks:
        #     doc = rag_api.recent_additional_chunks[doc_id]
        #     st.markdown(f'## {doc["doc_name"]}')
        #     st.write(f"Docs score: {doc['score']}")
        #     for idx, chunk_id in enumerate(doc["chunks"]):
        #         st.markdown(f"### Chunk {idx}")
        #         st.write(f"Content")
        #         # st.write(doc["chunks"][chunk_id].text[:max(len(doc["chunks"][chunk_id].text) // 3, 300)])
        #         st.write(doc["chunks"][chunk_id].text)
        #         st.write(f"Excerpt Page: {doc['chunks'][chunk_id].chunk_meta['excerpt_page_number']}")
        #         if doc["chunks"][chunk_id].chunk_meta.get("base_doc_id"):
        #             st.write(f"Base Document ID: {doc['chunks'][chunk_id].chunk_meta['base_doc_id']}")
        #     st.divider()
        write_source_docs(rag_api.recent_additional_docs, title="Additional Documents")

    st.session_state.messages.append(
        {
            "role": "AI",
            "content": response
        }
    )

        # TODO visualize pipline
