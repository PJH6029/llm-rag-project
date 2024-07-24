import streamlit as st
import os
from wasabi import msg
import boto3
import hashlib

from rag.type import *

def load_secrets():
    msg.info("Loading secrets...")
    keys = {
        "UPSTAGE_API_KEY",
        "OPENAI_API_KEY",
        "AWS_REGION",
        "KENDRA_INDEX_ID",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "S3_BUCKET_NAME",
        "KNOWLEDGE_BASE_OSS_ID",
    }
    for key in keys:
        os.environ[key] = st.secrets[key]
        
def merge_configs(*configs: dict) -> dict:
    merged_config = {}
    for config in configs:
        merged_config = {**merged_config, **config}
    return merged_config

def remove_none(config: dict) -> dict:
    return {k: v for k, v in config.items() if v is not None}

def remove_falsy(config: dict) -> dict:
    return {k: v for k, v in config.items() if v}

def get_presigned_url(s3_uri: str) -> str:
    s3 = boto3.client("s3")
    try:
        bucket, key = s3_uri.split("//")[1].split("/", 1)
        response = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
        return response
    except Exception as e:
        return None

def combine_chunks(chunks: list[Chunk], attach_url=False) -> list[CombinedChunks]:
    combined_chunks: dict[str, CombinedChunks] = {}
    for chunk in chunks:
        if chunk.doc_id not in combined_chunks:
            combined_chunks[chunk.doc_id] = CombinedChunks(doc_id=chunk.doc_id, doc_meta=chunk.doc_meta)
            if attach_url:
                combined_chunks[chunk.doc_id].link = get_presigned_url(chunk.doc_id)
        combined_chunks[chunk.doc_id].chunks.append(chunk)
        combined_chunks[chunk.doc_id].doc_mean_score += chunk.score

    for combined_chunk in combined_chunks.values():
        combined_chunk.doc_mean_score /= len(combined_chunk.chunks)
    
    combined_chunks_list = list(combined_chunks.values())
    combined_chunks_list = sorted(combined_chunks_list, key=lambda x: x.doc_mean_score, reverse=True)
    return combined_chunks_list

def format_chunks(chunks: list[Chunk]) -> str:
    combined_chunks = combine_chunks(chunks)
    context = {
        "base": "",
        "additional": "",
    }
    
    base_chunks = [chunk for chunk in combined_chunks if chunk.doc_meta.get("category") == "base"]
    additional_chunks = [chunk for chunk in combined_chunks if chunk.doc_meta.get("category") == "additional"]
    
    # base
    for combined_chunk in base_chunks:
        context["base"] += f"--- Document: {combined_chunk.doc_meta.get('doc_name', '')} ---\n"
        context["base"] += f"Average Score: {combined_chunk.doc_mean_score}\n"
        context["base"] += f"META:\n {combined_chunk.doc_meta}\n\n"
        for chunk in combined_chunk.chunks:
            context["base"] += f"{chunk.detail(doc_meta=False)}\n\n"
    
    # additional
    for combined_chunk in additional_chunks:
        context["additional"] += f"--- Document: {combined_chunk.doc_meta.get('doc_name', '')} ---\n"
        context["additional"] += f"Based on: {combined_chunk.doc_meta.get('base_doc_id', '')}\n"
        context["additional"] += f"Average Score: {combined_chunk.doc_mean_score}\n"
        context["additional"] += f"META:\n {combined_chunk.doc_meta}\n\n"
        for chunk in combined_chunk.chunks:
            context["additional"] += f"{chunk.detail(doc_meta=False)}\n\n"
    
    context_str = (
        "<base-context>\n"
        f"{context['base']}"
        "</base-context>\n"
        "<additional-context>\n"
        f"{context['additional']}"
        "</additional-context>"
    )
    return context_str

def format_history(history: list[ChatLog]) -> str:
    return "\n".join([f"{item['role'].upper()}: {item['content']}" for item in history])

def generate_id(input_string: str) -> str:
    hash_object = hashlib.sha256(input_string.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig