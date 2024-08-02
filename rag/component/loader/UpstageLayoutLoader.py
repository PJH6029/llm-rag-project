from typing import Iterator, Deque, Literal
from markdownify import markdownify as md
from wasabi import msg
import os
from collections import deque

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_upstage.layout_analysis import OutputType, SplitType

from rag import util

class UpstageLayoutLoader(BaseLoader):
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
    ) -> None:
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        
        self.source = self.file_path if source_type == "path" else self.file_name
        
        # to implement element overlap, split by element
        self.layout_loader = UpstageLayoutAnalysisLoader(
            file_path, output_type=anlaysis_output_type, split="element" if overlap_elem_size > 0 else "page", use_ocr=use_ocr
        )
        self.to_markdown = to_markdown
        self.cache_to_local = cache_to_local
        self.overlap_elem_size = overlap_elem_size
        self.backup_dir = backup_dir
        
        dir_name = os.path.splitext(os.path.basename(file_path))[0]
        if self.cache_to_local and os.path.exists(f"{self.backup_dir}/html/{dir_name}"):
            msg.warn(f"Backup directory already exists: {self.backup_dir}/<html or md>/{dir_name}")
    
    def _lazy_load_non_overlap(self) -> Iterator[Document]:
        for document in self.layout_loader.lazy_load():
            document = self._process(document)
            yield document
    
    def _process(self, document: Document) -> Document:
        file_name_with_ext = os.path.basename(self.file_path)
        file_name_without_ext = os.path.splitext(file_name_with_ext)[0]
        
        if self.cache_to_local:
            msg.info(f"Saving HTML into local: {file_name_without_ext}_{document.metadata.get('page')}.html")
            util.save_to_local(document, f"{self.backup_dir}/html/{file_name_without_ext}/{file_name_without_ext}_{document.metadata.get('page')}.html")
        
        if self.to_markdown:
            document.page_content = md(document.page_content)
            
            if self.cache_to_local:
                msg.info(f"Saving Markdown into local: {file_name_without_ext}_{document.metadata.get('page')}.md")
                util.save_to_local(document, f"{self.backup_dir}/markdown/{file_name_without_ext}/{file_name_without_ext}_{document.metadata.get('page')}.md")
        
        document.metadata["source"] = self.source
        
        return document
    
    def _lazy_load_overlap(self) -> Iterator[Document]:
        pprev_page_group = []
        prev_page_group = []
        current_page_group = []
        current_page = None
        first_trial = True
        
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
                    merged_doc = self._merge_elems(combined_elems, current_page - 1)
                    yield self._process(merged_doc)
                    
                pprev_page_group = prev_page_group
                prev_page_group = current_page_group
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
        
        # metadata except for page should be the same # TODO bounding_box, category(tagname), id are different
        # metadata = elem_docs[0].metadata
        metadata = {}
        metadata["page"] = page
        
        return Document(page_content=page_content, metadata=metadata)