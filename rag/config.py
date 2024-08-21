from typing import Optional
from pydantic import BaseModel, Field


class LanguageConfig(BaseModel):
    user: str = Field("Korean", description="User language")
    source: str = Field("English", description="Source language")
    assistant: str = Field("Korean", description="Assistant language")

class GlobalConfig(BaseModel):
    lang: LanguageConfig = LanguageConfig()
    context_hierarchy: bool = True
    
class RAGPipelineConfig:
    global_: Optional[GlobalConfig] = Field(None, alias="global", description="Global configuration")

class LoadConfig(BaseModel, RAGPipelineConfig):
    pass

class IngestionConfig(BaseModel, RAGPipelineConfig):
    ingestor: str = Field("pinecone-multivector", description="Ingestor name")
    embeddings: str = Field("text-embedding-3-small", description="Embeddings name")
    namespace: str = Field("parent", description="Pinecone namespace")
    sub_namespace: str = Field("child", description="Pinecone sub-namespace")

class TransformationEnableConfig(BaseModel):
    translation: bool = True
    rewriting: bool = True
    expansion: bool = False
    hyde: bool = True

class TransformationConfig(BaseModel, RAGPipelineConfig):
    model: str = Field("gpt-4o-mini", description="LLM model name")
    enable: TransformationEnableConfig = TransformationEnableConfig()
    
class RetrievalConfig(BaseModel, RAGPipelineConfig):
    retriever: list[str] = Field(["pinecone-multivector"], description="Retriever name")
    weights: Optional[list[float]] = Field(None, description="Retriever weights. If provided, should be the same length as retrievers")
    namespace: str = Field("parent", description="Pinecone namespace")
    sub_namespace: str = Field("child", description="Pinecone sub-namespace")
    embeddings: Optional[str] = Field("text-embedding-3-small", description="Embeddings name. If retriever does not require embeddings, this field is optional")
    top_k: int = Field(6, description="Top k results")

class GenerationConfig(BaseModel, RAGPipelineConfig):
    model: str = Field("gpt-4o", description="LLM model name")
    
class FactVerificationConfig(BaseModel, RAGPipelineConfig):
    model: str = Field("gpt-4o-mini", description="LLM model name")
    enable: bool = True

class RAGConfig(BaseModel):
    global_: GlobalConfig = Field(GlobalConfig(), alias="global")
    load: LoadConfig = LoadConfig()
    ingestion: IngestionConfig = IngestionConfig()
    transformation: TransformationConfig = TransformationConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    fact_verification: FactVerificationConfig = FactVerificationConfig()
