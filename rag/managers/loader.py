from wasabi import msg
from typing import Optional, Type, Iterable, Generator, Any, Union

from rag.type import *
from rag.managers.base import BasePipelineManager
from rag.component.loader import *

class LoaderManager(BasePipelineManager):
    def __init__(self) -> None:
        super().__init__()
        
        self.loaders: dict[str, Type[BaseRAGLoader]] = {
            "upstage_layout": UpstageLayoutLoader,
            "upstage_backup": UpstageLayoutBackupDirLoader,
        }
        self.selected_loader: Optional[BaseRAGLoader] = None
    
    def lazy_load_chunk(
        self,
        # resource_path: Any,
        *,
        loader: Optional[Union[BaseRAGLoader, BaseLoader]] = None,
        **resource_kwargs: Any,
    ) -> Iterable[Chunk]:
        if loader is None:
            loader = self._route_loader(**resource_kwargs)
        if isinstance(loader, BaseLoader) and not isinstance(loader, BaseRAGLoader):
            loader = BaseRAGLoader.from_lc_loader(loader)
        
        return loader.lazy_load_chunk()
    
    def load_chunk(
        self,
        # resource_path: Any,
        *,
        loader: Optional[BaseRAGLoader] = None,
        **resource_kwargs: Any,
    ) -> list[Chunk]:
        return list(self.lazy_load_chunk(loader=loader, **resource_kwargs))
    
    # TODO
    def _route_loader(self, **resource_kwargs) -> BaseRAGLoader:
        raise NotImplementedError("Route loader is not implemented yet")