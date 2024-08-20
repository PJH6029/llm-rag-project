from typing import Generator, Any, Optional
import streamlit as st
from wasabi import msg
import time

from frontend.util import *
from rag.api import api as rag_api
from rag.type import GenerationResult, VerificationResult

recent_fact_verification: Optional[VerificationResult] = None
doc_types = []

def generator_passthrough_with_cache(generator: Generator[GenerationResult, Any, Any]) -> Generator[str, Any, Any]:
    for item in generator:
        if item.get("generation") is not None:
            yield item.get("generation")
        elif item.get("fact_verification") is not None:
            fact_verification = item.get("fact_verification")

            global recent_fact_verification
            recent_fact_verification = fact_verification

def write_side_bar():
    if rag_api.recent_chunks is not None:
        with st.sidebar:
            st.markdown("# Source Documents")
            write_source_docs(rag_api.recent_chunks)

def run():
    global doc_types
    session_init(st.session_state)
    display_chat_history(st.session_state)
    
    if user_input := st.chat_input("Enter a message (줄임말은 정확도를 떨어뜨릴 수 있습니다)..."):  
        if "All" in doc_types:
            doc_types = rag_api.get_doc_types()
        print(doc_types)
        
        with st.chat_message("user"):
            st.session_state.messages.append(
                {
                    "role": "Human",
                    "content": user_input
                }
            )
            st.markdown(user_input)
            
        
        with st.chat_message("assistant"):
            query = st.session_state.messages[-1]["content"]
            rag_generator = rag_api.query_stream(
                query=query, history=st.session_state.translated_messages[:], doc_types=doc_types
            )
            # rag_generator = rag_api.fake_query_stream(
            #     query=query, history=st.session_state.translated_messages[:], doc_types=doc_types
            # )
            
            with st.status("Understaning question...") as status:
                start = time.time()
                transformed_queries = next(rag_generator).get("transformation")
                if transformed_queries:
                    write_queries(transformed_queries)
                else:
                    msg.warn("No transformation found. Check the query.")
                end = time.time()
                status.update(label=f"Understanding question... ({end-start:.2f} seconds)")
            
            with st.status("Looking for relevant contents...") as status:
                start = time.time()
                chunks = next(rag_generator).get("retrieval")
                if chunks:
                    write_chunks(chunks)
                else:
                    msg.warn("No data found. Check the retrieval result.")
                end = time.time()
                status.update(label=f"Looking for relevant contents... ({end-start:.2f} seconds)")
            
            response = st.write_stream(
                generator_passthrough_with_cache(rag_generator)
            )
            
            # Fact verification
            if recent_fact_verification is not None:
                label = "✅ Fact verified." if recent_fact_verification.verification else "❌ Fact not verified."
                with st.expander(label):
                    st.write(recent_fact_verification.reasoning)

        write_side_bar()

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

    else:
        write_side_bar()
    
    doc_types = st.multiselect(
        "Select document types to search",
        ["All"] + rag_api.get_doc_types(),
        default=["All"]
    )


