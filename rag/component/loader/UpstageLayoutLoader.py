from typing import Iterator, Deque, Literal, Optional, Callable
from markdownify import markdownify as md
from wasabi import msg
import os
import re
import json
from collections import deque

from langchain_core.documents import Document
from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_upstage.layout_analysis import OutputType, SplitType

from rag.component.loader.base import BaseRAGLoader
from rag.type import *
from rag import util

def get_total_pages(file_path: str) -> int:
    from PyPDF2 import PdfReader
    return len(PdfReader(file_path).pages)

def get_max_page_from_backup(backup_dir: str) -> int:
    max_page = -1
    for file in os.listdir(backup_dir):
        if not file.endswith(".html") and not file.endswith(".md"):
            continue
        file_name_without_ext = os.path.splitext(file)[0]
        page = int(file_name_without_ext.split("_")[-1])
        max_page = max(max_page, page)
    return max_page

# TODO load partial pages from backup, and the rest from analysis
class UpstageLayoutLoader(BaseRAGLoader):
    def __init__(
        self,
        file_path: str,
        anlaysis_output_type: OutputType | dict = "html",
        # split: SplitType = "page", # TODO make selectable
        overlap_elem_size: int = 0,
        use_ocr: bool | None = None,
        to_markdown: bool = False,
        cache_to_local: bool = False,
        backup_dir: str = "./layout_backup",
        source_type: Literal["path", "name"] = "path",
        metadata_ext: str = ".metadata.json",
        force_load: bool = False,
        *,
        metadata_handler: Optional[Callable[[dict], tuple[dict, dict]]] = None,
    ) -> None:
        super().__init__(metadata_handler=metadata_handler)
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        
        self.source = self.file_path if source_type == "path" else self.file_name
        
        # if backup file exists, use backup file instead
        self.layout_loader = None

        file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
        if to_markdown:
            backup_file_parent_dir_path = f"{backup_dir}/markdown/{file_name_without_ext}"
        else:
            backup_file_parent_dir_path = f"{backup_dir}/html/{file_name_without_ext}"
    
        max_page = -1
        total_pages = get_total_pages(file_path)
        if os.path.exists(backup_file_parent_dir_path):
            max_page = get_max_page_from_backup(backup_file_parent_dir_path)
            if max_page >= total_pages:
                # backup file exists
                msg.info(f"Backup file found: {backup_file_parent_dir_path}. Use backup file instead.")
                self.layout_loader = UpstageLayoutBackupLoader(
                    backup_file_parent_dir_path, metadata_handler=self.metadata_handler, metadata_ext=metadata_ext
                )
            else:
                # backup file exists, but not all pages are backed up
                msg.info(f"Backup file found: {backup_file_parent_dir_path}. But not all pages are backed up.")

        if self.layout_loader is None or force_load:
            # to implement element overlap, split by element
            self.layout_loader = UpstageLayoutAnalysisLoader(
                file_path, output_type=anlaysis_output_type, split="element" if overlap_elem_size > 0 else "page", use_ocr=use_ocr
            )
        # save to local only if layout loader is UpstageLayoutAnalysisLoader
        print(f"Use layout loader {self.layout_loader.__class__.__name__}")
        self.cache_to_local = cache_to_local & isinstance(self.layout_loader, UpstageLayoutAnalysisLoader)
        
        self.to_markdown = to_markdown
        self.overlap_elem_size = overlap_elem_size
        self.backup_dir = backup_dir
        self.metadata_ext = metadata_ext
        
        dir_name = os.path.splitext(os.path.basename(file_path))[0]
        if self.cache_to_local and os.path.exists(f"{self.backup_dir}/html/{dir_name}"):
            msg.warn(f"Backup directory already exists: {self.backup_dir}/<html or md>/{dir_name}")
    
    def _lazy_load_non_overlap(self) -> Iterator[Document]:
        # layout loader loads pages in order
        for document in self.layout_loader.lazy_load():
            document = self._process(document)
            yield document
    
    def _process(self, document: Document) -> Document:
        file_name_with_ext = os.path.basename(self.file_path)
        file_name_without_ext = os.path.splitext(file_name_with_ext)[0]
        
        # TODO expand to general metadata handler
        metadata_to_dump = document.metadata.copy()
        metadata_to_dump["source"] = self.source
        persistent_metadata = util.persistent_metadata_handler(metadata_to_dump)[0]
        metadata_to_dump.update(persistent_metadata)
        
        if self.cache_to_local:
            msg.info(f"Saving HTML into local: {file_name_without_ext}_{document.metadata.get('page')}.html")
            util.save_to_local(document.page_content, f"{self.backup_dir}/html/{file_name_without_ext}/{file_name_without_ext}_{document.metadata.get('page')}.html")
            util.save_to_local(json.dumps(metadata_to_dump), f"{self.backup_dir}/html/{file_name_without_ext}/{file_name_without_ext}_{document.metadata.get('page')}.html{self.metadata_ext}")
                
        if self.to_markdown:
            document.page_content = util.markdownify(document.page_content)
            
            if self.cache_to_local:
                msg.info(f"Saving Markdown into local: {file_name_without_ext}_{document.metadata.get('page')}.md")
                util.save_to_local(document.page_content, f"{self.backup_dir}/markdown/{file_name_without_ext}/{file_name_without_ext}_{document.metadata.get('page')}.md")
                util.save_to_local(json.dumps(metadata_to_dump), f"{self.backup_dir}/markdown/{file_name_without_ext}/{file_name_without_ext}_{document.metadata.get('page')}.md{self.metadata_ext}")
        
        document.metadata["source"] = self.source
        
        return document
    
    def _lazy_load_overlap(self) -> Iterator[Document]:
        pprev_page_group = []
        pprev_page = None
        prev_page_group = []
        prev_page = None
        current_page_group = []
        current_page = None
        first_trial = True
        
        # layout loader loads elements in order
        for elem_doc in self.layout_loader.lazy_load():
            page = elem_doc.metadata.get("page")
            if page is None:
                msg.warn("Page number not found in metadata. Skipping.")
                continue
            
            if current_page is None:
                current_page = page
                
            if page != current_page:
                if first_trial:
                    # skip the first trial
                    first_trial = False
                else:
                    combined_elems = pprev_page_group[-self.overlap_elem_size:] + prev_page_group + current_page_group[:self.overlap_elem_size]
                    merged_doc = self._merge_elems(combined_elems, prev_page)
                    yield self._process(merged_doc)
                    
                pprev_page_group = prev_page_group
                pprev_page = prev_page
                prev_page_group = current_page_group
                prev_page = current_page
                current_page_group = []
            
            current_page_group.append(elem_doc)
            current_page = page
        
        if current_page > 1:
            # second last page
            # skipt if only one page
            combined_elems = pprev_page_group[-self.overlap_elem_size:] + prev_page_group + current_page_group[:self.overlap_elem_size]
            merged_doc = self._merge_elems(combined_elems, current_page - 1)
            yield self._process(merged_doc)
        
        # last page
        combined_elems = prev_page_group[-self.overlap_elem_size:] + current_page_group
        merged_doc = self._merge_elems(combined_elems, current_page)
        yield self._process(merged_doc)
    
    def lazy_load(self) -> Iterator[Document]:
        if self.overlap_elem_size > 0:
            return self._lazy_load_overlap()
        else:
            return self._lazy_load_non_overlap()
        
    def _merge_elems(self, elem_docs: list[Document], page: int) -> Document:
        page_content = " ".join(
            [elem_doc.page_content for elem_doc in elem_docs]
        )
        
        metadata = {}
        metadata["page"] = page
        
        return Document(page_content=page_content, metadata=metadata)

class UpstageLayoutBackupLoader(BaseRAGLoader):
    def __init__(
        self,
        backup_file_path: str,
        metadata_handler: Optional[Callable[[dict], tuple[dict, dict]]] = None,
        metadata_ext: str = ".metadata.json",
    ) -> None:
        super().__init__(metadata_handler)
        self.metadata_json_ext = metadata_ext
        self.backup_file_path = backup_file_path
    
    def _doc_from_backup_page(self, page_file_path: str) -> Document:
        file_name = os.path.basename(page_file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        
        page = int(file_name_without_ext.split("_")[-1])
        if page <= 0:
            msg.warn(f"Page from file path {page_file_path} is less than 1: {page}.")
            raise ValueError(f"Page from file path {page_file_path} is less than 1: {page}.")
        
        pdf_file_name = "_".join(file_name_without_ext.split("_")[:-1]) + ".pdf"
        
        doc_id = pdf_file_name
        
        with open(page_file_path, "r") as f:
            content = f.read()
        
        # TODO redundant?
        metadata_path = f"{page_file_path}{self.metadata_json_ext}"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        document = Document(
            page_content=content,
            metadata={
                "doc_id": doc_id,
                "page": page,
                **metadata
            }
        )

        return document
    
    def lazy_load(self) -> Iterator[Document]:
        files = filter(lambda file: file.endswith(".html") or file.endswith(".md"), os.listdir(self.backup_file_path))
        page_extractor = lambda file: int(os.path.splitext(file)[0].split("_")[-1])
        for file in sorted(files, key=page_extractor):
        # for file in os.listdir(self.backup_file_path):
            if file.endswith(".html") or file.endswith(".md"):
                yield self._doc_from_backup_page(f"{self.backup_file_path}/{file}")

class UpstageLayoutBackupDirLoader(BaseRAGLoader):
    def __init__(
        self, 
        backup_dir: str,
        metadata_handler: Optional[Callable[[dict], tuple[dict, dict]]] = None, 
        metadata_ext: str = ".metadata.json",
    ) -> None:
        super().__init__(metadata_handler)
        self.metadata_json_ext = metadata_ext
        self.backup_dir = backup_dir
        
        html_dir = f"{backup_dir}/html"
        md_dir = f"{backup_dir}/markdown"
        
        if os.path.exists(md_dir):
            self.data_source_dir = md_dir
            self.data_source_ext = ".md"
        else:
            self.data_source_dir = html_dir
            self.data_source_ext = ".html"
            
    def lazy_load(self) -> Iterator[Document]:
        for root, _, files in os.walk(self.data_source_dir):
            # if one of files contains the data source extension, iterate roo
            if any(file.endswith(self.data_source_ext) for file in files):
                loader = UpstageLayoutBackupLoader(root, metadata_handler=self.metadata_handler, metadata_ext=self.metadata_json_ext)
                yield from loader.lazy_load()
