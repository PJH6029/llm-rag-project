from typing import Optional, Type, Iterable, Callable
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

from rag.type import *
from rag import util

class BaseRAGLoader(BaseLoader):
    def __init__(
        self,
        metadata_handler: Optional[Callable[[dict], tuple[dict, dict]]] = None,
        *,
        lc_loader: Optional[BaseLoader] = None,
    ) -> None:
        super().__init__()
        self.metadata_handler = metadata_handler
        
        # intantiate from langchain loader
        self.lc_loader = lc_loader
        if lc_loader is not None:
            self.__dict__.update(lc_loader.__dict__)
            self._wrap_methods()

    def _wrap_methods(self) -> None:
        if self.lc_loader is None:
            return
        for method_name in dir(self.lc_loader):
            method = getattr(self.lc_loader, method_name)
            if callable(method) and not method_name.startswith("__"):
                wrapped_method = self._wrap_method(method)
                setattr(self, method_name, wrapped_method)
    
    def _wrap_method(self, method: Callable) -> Callable:
        def wrapped_method(*args, **kwargs):
            return method(*args, **kwargs)
        return wrapped_method

    def lazy_load_chunk(self) -> Iterable[Chunk]:
        for document in self.lazy_load():
            yield util.doc_to_chunk(document, metadata_handler=self.metadata_handler)
    
    def load_chunk(self) -> list[Chunk]:
        return list(self.lazy_load_chunk())
    
    @classmethod
    def from_lc_loader(
        cls, 
        loader: BaseLoader,
        metadata_handler: Optional[Callable[[dict], tuple[dict, dict]]] = None,
    ) -> "BaseRAGLoader":
        return cls(metadata_handler=metadata_handler, lc_loader=loader)
    
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
        parsed = urlparse(url)
        return bool(parsed.scheme) and bool(parsed.netloc)
