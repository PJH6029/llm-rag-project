import contextlib
import os
from wasabi import msg
import boto3
import hashlib
from typing import Optional, Any, Iterable, Callable, Generator, Type
import uuid
import json
import time
import re

import streamlit as st
from markdownify import markdownify as md

from rag.type import *
from rag.config import *

def load_secrets():
    msg.info("Loading secrets...")
    for key in st.secrets.keys():
        try:
            os.environ[key] = st.secrets[key]
        except KeyError:
            msg.warn(f"Secret '{key}' not found. You may not be able to access some features.")

def load_config(config_path = "config/config.json") -> dict:
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        rag_config = config.get("rag", {})
        global_config = config.get("global", {})
        rag_config.get("global", {}).update(global_config)
        return rag_config
    except Exception as e:
        msg.fail(f"Failed to load config: {e}")
        return {}

# def merge_configs(*configs: dict) -> dict:
#     merged_config = {}
#     for config in configs:
#         merged_config = {**merged_config, **config}
#     return merged_config

def attach_global_config(config: RAGPipelineConfig, global_config: GlobalConfig) -> RAGPipelineConfig:
    config.global_ = global_config
    return config

@contextlib.contextmanager
def time_logger(
    start_msg_cb: Callable[..., str],
    end_msg_cb: Callable[..., str],
) -> Generator[None, None, None]:
    start = time.time()
    msg.info(start_msg_cb())
    yield
    end = time.time()
    msg.good(f"{end_msg_cb()} ({end-start:.2f}s)")

def remove_none(config: dict) -> dict:
    return {k: v for k, v in config.items() if v is not None}

def remove_falsy(
    config: dict,
    falsy_values: dict[Type, list[Any]] = {
        int: [-1],
    }
) -> dict:
    result = {}
    for k, v in config.items():
        if v is None:
            continue
        if type(v) in falsy_values:
            # look up falsy values
            if v not in falsy_values[type(v)]:
                result[k] = v
        else:
            # default falsy values
            if bool(v):
                result[k] = v    
    return result

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
        combined_chunks[chunk.doc_id].doc_max_score = max(combined_chunks[chunk.doc_id].doc_max_score, chunk.score)

    for combined_chunk in combined_chunks.values():
        combined_chunk.doc_mean_score /= len(combined_chunk.chunks)
    
    combined_chunks_list = list(combined_chunks.values())
    combined_chunks_list = sorted(combined_chunks_list, key=lambda x: x.doc_mean_score, reverse=True)
    combined_chunks_list = sorted(combined_chunks_list, key=lambda x: x.doc_max_score, reverse=True)
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

def format_chunks_hierarchy_context(chunks: list[Chunk], ascending_additional: bool=True) -> str:
    combined_chunks = combine_chunks(chunks)
    context = {
        "base": "",
        "additional": "",
    }
    
    base_chunks = [chunk for chunk in combined_chunks if chunk.doc_meta.get("doc_type") == "base"]
    additional_chunks = [chunk for chunk in combined_chunks if chunk.doc_meta.get("doc_type") == "additional"]
    
    # base
    context["base"] = _format_combined_chunks(base_chunks)
    
    # additional
    additional_chunks.sort(key=lambda x: x.doc_max_score, reverse=not ascending_additional) # resolve lost in middle problem
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

def flatten_dict(d: dict, ignore_dup: bool=False, parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, ignore_dup, "" if ignore_dup else new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def deflatten_dict(d: dict[str, Any], sep: str = "/") -> dict:
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_ptr = result
        for part in parts[:-1]:
            d_ptr = d_ptr.setdefault(part, {})
        d_ptr[parts[-1]] = value
    return result

def _default_metadata_handler(metadata: dict) -> tuple[dict, dict]:
    doc_id = MetadataSearch.search_doc_id(metadata)
    doc_source = MetadataSearch.search_source(metadata)
    if doc_id is None and doc_source is None:
        raise ValueError("doc_id & doc_source not found in metadata")

    # if one of them is None, use the other
    if doc_id is None:
        doc_id = doc_source
    elif doc_source is None:
        doc_source = doc_id
    
    page = metadata.pop("page", -1)
    doc_meta = {
        **metadata,
        "doc_id": doc_id,
        "source": doc_source,
    }
    chunk_meta = {
        "page": page,
    }
    return doc_meta, chunk_meta

def doc_to_chunk(
    document: Document,
    *,
    metadata_handler: Optional[Callable[[dict], tuple[dict, dict]]] = None,
) -> Chunk:
    if metadata_handler is None:
        final_metadata_handler = _default_metadata_handler
    else:
        # chaining metadata handler
        def final_metadata_handler(metadata: dict) -> tuple[dict, dict]:
            doc_meta, chunk_meta = metadata_handler(metadata)
            default_doc_meta, default_chunk_meta = _default_metadata_handler(metadata)
            doc_meta = {**default_doc_meta, **doc_meta}
            chunk_meta = {**default_chunk_meta, **chunk_meta}
            return doc_meta, chunk_meta
    try:
        doc_meta, chunk_meta = final_metadata_handler(document.metadata)
    except Exception as e:
        msg.warn(f"Failed to handle metadata: {e}. Using default metadata handler.")
        doc_meta, chunk_meta = _default_metadata_handler(document.metadata)

    chunk_id = str(uuid.uuid4())
    chunk = Chunk(
        text=document.page_content,
        doc_id=doc_meta["doc_id"],
        chunk_id=chunk_id,
        doc_meta=remove_falsy(doc_meta),
        chunk_meta=remove_falsy(chunk_meta)
    )
    return chunk

def is_in_nested_keys(d: dict, key: str) -> bool:
    if key in d:
        return True
    for k, v in d.items():
        if isinstance(v, dict):
            if is_in_nested_keys(v, key):
                return True
    return False

def validate_metadata(metadata: dict) -> bool:
    # forbidden characters for metadata keys
    forbidden_chars = set([".", "#", "[", "]", "/"])
    for key, value in metadata.items():
        if set(key) & forbidden_chars:
            return False
        if isinstance(value, dict):
            if not validate_metadata(value):
                return False
    return True

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
    
    if not os.path.exists(file_path):
        msg.fail(f"File not found: {file_path}")
        return False
    
    if not os.path.exists(metadata_file_path):
        if metadata is None:
            msg.warn(f"Metadata file not found: {metadata_file_path}, and metadata is not provided. Uploading file only.")
        else:
            # write metadata to file
            msg.info(f"Writing metadata to file: {metadata_file_path}")
            with open(metadata_file_path, "w") as f:
                f.write(json.dumps(metadata, indent=4))
    else:
        with open(metadata_file_path, "r") as f:
            metadata = json.load(f)
    
    if not validate_metadata(metadata):
        msg.fail(f"Invalid metadata: {metadata}")
        return False
    
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

def save_to_local(content: Any, file_path: str) -> bool:
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    
    try:
        with open(file_path, "w") as f:
            f.write(content)
    except Exception as e:
        msg.fail(f"Failed to save content to local: {e}")
        return False
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

def markdownify(text: str, **markdownify_options) -> str:
    text = md(text, **markdownify_options).replace("\xa0", " ").strip()
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", text)
    return cleaned_text