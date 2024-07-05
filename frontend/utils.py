from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
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

# def to_sync_generator(async_gen: AsyncGenerator):
#     while True:
#         try:
#             yield asyncio.run(anext(async_gen))
#         except StopAsyncIteration:
#             break