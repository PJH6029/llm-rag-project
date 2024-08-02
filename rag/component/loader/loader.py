from typing import Iterable, Optional
import os

from rag.type import *
from rag.component.loader import PDFWithMetadataLoader, UpstageLayoutLoader
from rag import util

def lazy_load(file_path: str) -> Iterable[Chunk]:
    """Load chunk from s3 or local file path

    Args:
        file_path (str): s3 url or local file path

    Returns:
        Iterable[Chunk]: chunk iterator
    """
    chunks_iter = PDFWithMetadataLoader(
        file_path,
        loader=UpstageLayoutLoader,
        loader_kwargs={
            "use_ocr": True,
            "to_markdown": True,
            "overlap_elem_size": 2,
            "cache_to_local": True,
            "backup_dir": "./layout_overlap_backup", # TODO configurable
        }
    ).lazy_load_as_chunk()
    return chunks_iter

def lazy_load_from_backup(backup_dir: str, object_location: Optional[str] = None) -> Iterable[Chunk]:
    html_dir = f"{backup_dir}/html"
    md_dir = f"{backup_dir}/markdown"

    # if md_dir exists, load from md_dir
    if os.path.exists(md_dir):
        data_source_dir = md_dir
        data_source_ext = ".md"
    else:
        data_source_dir = html_dir
        data_source_ext = ".html"

    for root, _, files in os.walk(data_source_dir):
        for file in files:
            if not file.endswith(data_source_ext):
                continue
            file_path = os.path.join(root, file)
            chunk = _chunk_from_backup_page(file_path, object_location)
            yield chunk


def _chunk_from_backup_page(page_file_path: str, object_location: Optional[str]=None) -> Chunk:
    file_name = os.path.basename(page_file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    
    if object_location is not None:
        file_object_key = os.path.join(object_location, file_name)
        doc_id = util.get_s3_url_from_object_key(file_object_key)
    else:
        doc_id = page_file_path
    
    page = int(file_name_without_ext.split("_")[-1])
    with open(page_file_path, "r") as f:
        content = f.read()
    
    document = Document(
        page_content=content,
        metadata={
            "doc_id": doc_id,
            "source": doc_id,
            "page": page
        }
    )
    
    return util.doc_to_chunk(document)