import streamlit as st
import os
from wasabi import msg
import boto3
import hashlib
from typing import Optional, Any, Iterable, Callable
import uuid
import json

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
    }
    for key in keys:
        try:
            os.environ[key] = st.secrets[key]
        except KeyError:
            msg.warn(f"Secret '{key}' not found. You may not be able to access some features.")
        
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

def _format_combined_chunks(combined_chunks: list[CombinedChunks]) -> str:
    result = ""
    for combined_chunk in combined_chunks:
        result += f"--- Document: {combined_chunk.doc_meta.get('doc_name', '')} ---\n"
        if combined_chunk.doc_meta.get("base_doc_id"):
            result += f"Based on: {combined_chunk.doc_meta.get('base_doc_id')}\n"
        result += f"Average Score: {combined_chunk.doc_mean_score}\n"
        result += f"DOC META:\n {combined_chunk.doc_meta}\n\n"
        for chunk in combined_chunk.chunks:
            result += f"{chunk.detail(doc_meta=False)}\n\n"
    return result

def format_chunks_single_context(chunks: list[Chunk]) -> str:
    combined_chunks = combine_chunks(chunks)
    
    return _format_combined_chunks(combined_chunks)

# TODO resolve lost in middle problem. Make high score base chunks and additional chunks physically close to each other.
def format_chunks_hierarchy_context(chunks: list[Chunk]) -> str:
    combined_chunks = combine_chunks(chunks)
    context = {
        "base": "",
        "additional": "",
    }
    
    base_chunks = [chunk for chunk in combined_chunks if chunk.doc_meta.get("category") == "base"]
    additional_chunks = [chunk for chunk in combined_chunks if chunk.doc_meta.get("category") == "additional"]
    
    # base
    context["base"] = _format_combined_chunks(base_chunks)
    
    # additional
    context["additional"] = _format_combined_chunks(additional_chunks)
    
    context_str = (
        "<base-context>\n"
        f"{context['base']}"
        "</base-context>\n"
        "<additional-context>\n"
        f"{context['additional']}"
        "</additional-context>"
    )
    return context_str

def format_chunks(chunks: list[Chunk], use_hierarchy=False) -> str:
    if use_hierarchy:
        return format_chunks_hierarchy_context(chunks)
    else:
        return format_chunks_single_context(chunks)

def format_history(history: list[ChatLog]) -> str:
    return "\n".join([f"{item['role'].upper()}: {item['content']}" for item in history])

def generate_id(input_string: str) -> str:
    hash_object = hashlib.sha256(input_string.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig

# TODO separating with underscores might occur bugs if the key end with trailing underscores
def flatten_dict(d: dict, ignore_dup: bool=False, parent_key: str = "", sep: str = "___") -> dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, ignore_dup, "" if ignore_dup else new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def deflatten_dict(d: dict[str, Any], sep: str = "___") -> dict:
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_ptr = result
        for part in parts[:-1]:
            d_ptr = d_ptr.setdefault(part, {})
        d_ptr[parts[-1]] = value
    return result

def doc_to_chunk(document: Document) -> Chunk:
    try:
        chunk_id = str(uuid.uuid4())
        page = document.metadata.pop("page", -1)
        
        chunk = Chunk(
            text=document.page_content,
            doc_id=MetadataSearch.search_doc_id(document.metadata),
            chunk_id=chunk_id,
            doc_meta={
                **document.metadata,
                "source": MetadataSearch.search_source(document.metadata),
            },
            chunk_meta={
                "chunk_id": chunk_id,
                "page": page,
            }
        )
        return chunk
    except KeyError as e:
        raise ValueError(f"doc_id not found in document metadata: {e}")

def is_in_nested_keys(d: dict, key: str) -> bool:
    if key in d:
        return True
    for k, v in d.items():
        if isinstance(v, dict):
            if is_in_nested_keys(v, key):
                return True
    return False

def get_s3_url_from_object_key(object_key: str) -> str:
    return f"s3://{os.environ['S3_BUCKET_NAME']}/{object_key}"

def upload_to_s3(
    file_path: str, 
    object_key: Optional[str]=None,
    content_type: Optional[str]=None,
) -> bool:
    if object_key is None:
        object_key = os.path.basename(file_path)
    
    try:
        s3 = boto3.client("s3")
        if content_type:
            s3.upload_file(file_path, os.environ["S3_BUCKET_NAME"], object_key, ExtraArgs={
                "ContentType": content_type,
            })
        else:
            s3.upload_file(file_path, os.environ["S3_BUCKET_NAME"], object_key)
    except Exception as e:
        msg.fail(f"Failed to upload file to S3: {e}")
        return False
    # msg.good(f"File uploaded to S3: {object_key}")
    return True

def upload_to_s3_with_metadata(
    file_path: str,
    object_location: str = "",
    metadata: Optional[dict] = None,
    metadata_ext: str = ".metadata.json",
) -> bool:
    file_name = os.path.basename(file_path)
    metadata_file_name = f"{file_name}{metadata_ext}"

    file_directory = os.path.dirname(file_path)
    metadata_file_path = os.path.join(file_directory, metadata_file_name)
    
    # if file or metadata does not exist, return
    if not os.path.exists(file_path):
        msg.fail(f"File not found: {file_path}")
        return False
    
    if not os.path.exists(metadata_file_path):
        if metadata is None:
            msg.fail(f"Metadata file not found: {metadata_file_path}, and metadata is not provided.")
            return False
        # write metadata to file
        msg.info(f"Writing metadata to file: {metadata_file_path}")
        with open(metadata_file_path, "w") as f:
            f.write(json.dumps(metadata, indent=4))
    
    file_object_key = os.path.join(object_location, file_name)
    metadata_object_key = os.path.join(object_location, metadata_file_name)

    upload_to_s3(file_path, file_object_key, content_type="application/pdf")
    upload_to_s3(metadata_file_path, metadata_object_key, content_type="application/json")
    return True

def get_presigned_url_upload(file_name):
    pass

def download_from_s3(object_key_or_url: str, file_path: str) -> bool:
    if object_key_or_url.startswith("s3://"):
        bucket, key = split_s3_url(object_key_or_url)
    else:
        bucket = os.environ["S3_BUCKET_NAME"]
        key = object_key_or_url
    
    try:
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, file_path)
    except Exception as e:
        msg.fail(f"Failed to download file from S3: {e}")
        return False
    # msg.good(f"File downloaded from S3: {object_key_or_url}, to: {file_path}")
    return True

def split_s3_url(s3_url: str) -> tuple[str, str]:
    try:
        bucket, key = s3_url.split("//")[1].split("/", 1)
        return bucket, key
    except Exception as e:
        msg.fail(f"Failed to split S3 URL: {e}")
        return None, None

def save_to_local(document: Document, file_path: str) -> bool:
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    
    try:
        with open(file_path, "w") as f:
            f.write(document.page_content)
    except Exception as e:
        msg.fail(f"Failed to save document to local: {e}")
        return False
    # msg.good(f"Document saved to local: {file_path}")
    return True

def execute_as_batch(
    iterable: Iterable[Any], 
    batch_size: int = 10,
    func: Optional[Callable[[list[Any]], Any]] = None,
) -> int:
    batch = []
    item_cnt = 0
    for item in iterable:
        batch.append(item)
        item_cnt += 1
        if len(batch) == batch_size:
            if func:
                func(batch)
            batch = []
    
    if batch:
        if func:
            func(batch)
    
    return item_cnt

class MetadataSearch:
    @staticmethod
    def search_doc_id(metadata: dict) -> Optional[str]:
        doc_id = metadata.get("doc_meta", {}).get("doc_id")
        if doc_id:
            return doc_id
        
        doc_id = metadata.get("doc_id")
        if doc_id:
            return doc_id
        
        return None
    
    @staticmethod
    def search_chunk_id(metadata: dict) -> Optional[str]:
        chunk_id = metadata.get("chunk_meta", {}).get("chunk_id")
        if chunk_id:
            return chunk_id
        
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return chunk_id
        
        return None
    
    @staticmethod
    def search_source(metadata: dict) -> Optional[str]:
        source = metadata.get("doc_meta", {}).get("source")
        if source:
            return source
        
        source = metadata.get("source")
        if source:
            return source
        
        return MetadataSearch.search_doc_id(metadata)
    
def flatten_queries(queries: TransformationResult) -> list[str]:
    result = []
    for k in queries:
        if isinstance(queries[k], list):
            result.extend(queries[k])
        else:
            result.append(queries[k])
    return result