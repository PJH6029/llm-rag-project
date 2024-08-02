from typing import Literal, Union, get_args
from pydantic import BaseModel

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

class Chunk(BaseModel):
    text: str
    doc_id: str
    chunk_id: str
    doc_meta: dict = {}
    chunk_meta: dict = {}
    score: float = 0.0
    source_retriever: str = ""
    
    def __str__(self) -> str:
        return f"Chunk(doc_id={self.doc_id}, chunk_id={self.chunk_id}, score={self.score}, doc_meta={self.doc_meta}, chunk_meta={self.chunk_meta})"

    def __repr__(self) -> str:
        return self.__str__()
    
    def detail(self, doc_meta=True) -> str:
        return (
            f"--- Chunk: {self.chunk_id} ---\n"
            f"Score: {self.score}\n"
            f"TEXT:\n {self.text}\n"
            f"CHUNK META:\n {self.chunk_meta}\n"
        ) + (f"DOC META:\n {self.doc_meta}" if doc_meta else "")
    
    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "chunk_id": self.chunk_id,
                "doc_id": self.doc_id,
                "doc_meta": self.doc_meta,
                "chunk_meta": self.chunk_meta,
            }
        )


class CombinedChunks(BaseModel):
    chunks: list[Chunk] = []
    doc_id: str = ""
    doc_meta: dict = {}
    doc_mean_score: float = 0.0
    link: str = ""
    

FilterOp = Literal[
    "equals", "notEquals", "greaterThan", "greaterThanOrEquals",
    "lessThan", "lessThanOrEquals", "in", "notIn", "startsWith",
]
filter_ops = get_args(FilterOp)

FilterValue = Union[str, int, float, list[str], bool]

LogicalOp = Literal["andAll", "orAll"]
logical_ops = get_args(LogicalOp)

valid_ops = filter_ops + logical_ops

class FilterPredicate(BaseModel):
    op: FilterOp
    key: str
    value: FilterValue
    
    def dict(self):
        return {self.op: {"key": self.key, "value": self.value}}
    
class FilterExpression(BaseModel):
    op: LogicalOp
    predicates: list[Union[FilterPredicate, "FilterExpression"]]
    
    def dict(self):
        return {self.op: [predicate.dict() for predicate in self.predicates]}

Filter = Union[FilterPredicate, FilterExpression]

TransformationResult = dict[Literal["translation", "expansion", "rewriting", "hyde"], Union[str, list[str]]]

ChatLog = dict[Literal["role", "content"], str]
GenerationResult = dict[Literal["transformation", "retrieval", "generation", "fact_verification"], Union[TransformationResult, list[Chunk], str]]
 
AnyLanguageModel = Union[BaseLanguageModel]
AnyEmbeddings = Union[Embeddings]