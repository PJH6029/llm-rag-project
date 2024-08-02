from langchain_core.prompts import ChatPromptTemplate

summarize_prompt_template = """
Summarize the following chunk, considering the metadata and content of the chunk.
Metadata contains the two types of information: document-level metadata and chunk-level metadata.
Document-level metadata contains information about the source document itself, such as the document ID, document name, category, version, and URI.
Chunk-level metadata contains information about the chunk itself, such as the chunk ID and score.

You don't have to include any additional information in your answer. Just summarize the chunk.

Chunk content:
{text}

Document-level metadata:
{doc_meta}

Chunk-level metadata:
{chunk_meta}
"""
summarize_prompt = ChatPromptTemplate.from_template(summarize_prompt_template)

hypothetical_queries_template = """
For the following chunk with metadata, generate {n} hypothetical queries that a user might ask.
Metadata contains the two types of information: document-level metadata and chunk-level metadata.
Document-level metadata contains information about the source document itself, such as the document ID, document name, category, version, and URI.
Chunk-level metadata contains information about the chunk itself, such as the chunk ID and score.

You don't have to include any additional information in your answer. 
Just generate hypothetical queries.
All queries should be separated by a newline, and end with a question mark.

Chunk content:
{text}

Document-level metadata:
{doc_meta}

Chunk-level metadata:
{chunk_meta}
"""
hypothetical_queries_prompt = ChatPromptTemplate.from_template(hypothetical_queries_template)