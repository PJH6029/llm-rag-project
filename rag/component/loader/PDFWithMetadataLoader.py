from typing import Union, Optional, Iterator, Type, Any
from pathlib import Path
from urllib.parse import urlparse
import tempfile
import os, json
from wasabi import msg
import uuid

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from rag.type import *
from rag import util

class PDFWithMetadataLoader(BaseLoader):
    def __init__(
        self, 
        file_path: str,
        *,
        loader: Optional[Type[BaseLoader]] = None,
        loader_kwargs: dict[str, Any] = {},
        metadata_ext: str = ".metadata.json",
    ) -> None:
        self.file_path = file_path
        # TODO handling specified location for metadata
        # currently, metadata is assumed to be in the same location as the file
        self.metadata_path = f"{self.file_path}{metadata_ext}"
        
        self.file_name = os.path.basename(self.file_path)
        self.metadata_name = f"{self.file_name}{metadata_ext}"
        
        if self._is_s3_url(self.file_path):        
            # download metadata to a temporary location
            self.temp_dir = tempfile.TemporaryDirectory()
            temp_file_path = os.path.join(self.temp_dir.name, self.file_name)
            temp_metadata_path = os.path.join(self.temp_dir.name, self.metadata_name)
            util.download_from_s3(self.file_path, temp_file_path)
            util.download_from_s3(self.metadata_path, temp_metadata_path)
            
            self.source = self.file_path
            file_path_for_loader = temp_file_path
            metadata_path_for_loader = temp_metadata_path
        else:
            self.source = self.file_name # if not s3, use file name as source
            file_path_for_loader = self.file_path
            metadata_path_for_loader = self.metadata_path
        
        if loader is None:
            msg.warn("Loader not specified. Fallback to PyPDFLoader.")
            self.loader = PyPDFLoader(file_path=file_path_for_loader) # fallback to PyPDFLoader
        else:
            msg.info(f"Using loader: {loader.__name__}")
            self.loader = loader(
                **{**loader_kwargs, "file_path": file_path_for_loader}
            )
        
        try:
            # load metadata
            with open(metadata_path_for_loader, "r") as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = {}
            
    def lazy_load(self) -> Iterator[Document]:
        for document in self.loader.lazy_load():
            document.metadata["source"] = self.source # overwrite or add source
            document.metadata.update(self.metadata)
            
            if not util.is_in_nested_keys(self.metadata, "doc_id"):
                document.metadata["doc_id"] = document.metadata["source"]

            yield document
    
    def lazy_load_as_chunk(self) -> Iterator[Chunk]:
        """Yield a single chunk from the document

        Yields:
            Iterator[Chunk]: A single chunk
        """
        for document in self.loader.lazy_load():
            document.metadata["source"] = self.source # overwrite or add source
            document.metadata.update(self.metadata)
            
            if not util.is_in_nested_keys(self.metadata, "doc_id"):
                document.metadata["doc_id"] = document.metadata["source"]
            
            chunk = util.doc_to_chunk(document)
            yield chunk
        
    def load_as_chunk(self) -> list[Chunk]:
        return list(self.lazy_load_as_chunk())

    @staticmethod
    def _is_s3_url(url: str) -> bool:
        try:
            result = urlparse(url)
            if result.scheme == "s3" and result.netloc:
                return True
            return False
        except ValueError:
            return False

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)