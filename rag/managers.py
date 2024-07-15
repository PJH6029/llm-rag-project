from wasabi import msg
import time
import tiktoken

from langchain.prompts import ChatPromptTemplate

from rag.interfaces import *
from rag.retrieve.KnowledgeBaseRetriever import KnowledgeBaseRetriever
from rag.types import *
from rag.generate.GPT4Generator import GPT4Generator
from rag.generate.GPT3Generator import GPT3Generator
from rag.retrieve.KendraRetriever import KendraRetriever
from rag.retrieve.FAISSRetriever import FAISSRetriever
from rag.revise.GPTRevisor import GPTRevisor
from rag import utils
from rag.prompts import hyde_prompt


class ChunkerManager:
    def __init__(self) -> None:
        self.chunkers: dict[str, Chunker] = {
            # TODO
        }
        self.selected_chunker: str = None

    def chunk(self, documents: list[Document]) -> list[Document]:
        msg.info(f"Starting chunking with {self.selected_chunker}")
        start_time = time.time()

        chunker = self.get_selected_chunker()
        chunked_documents = chunker.chunk(documents)
        msg.info(f"Chunking completed. Starting validation...")

        if self.validate_chunks(chunked_documents):
            msg.good(f"Chunking completed with {sum([len(doc.chunks) for doc in chunked_documents])} in {time.time() - start_time:.2f} seconds")
            return chunked_documents
        else:
            msg.warn("Validating chunks failed.")
            return []
    
    def add_metadata(self, documents: list[Document]) -> list[Document]:
        # TODO
        return documents

    def set_chunker(self, chunker_name: str) -> bool:
        if chunker_name in self.chunkers:
            msg.info(f"Setting chunker to {chunker_name}")
            self.selected_chunker = chunker_name
            return True
        else:
            msg.warn(f"Chunker {chunker_name} not found. Setting chunker to None")
            self.selected_chunker = None
            return False
    
    def get_selected_chunker(self) -> Chunker:
        return self.chunkers.get(self.selected_chunker, None)

    def validate_chunks(self, documents: list[Document], max_tokens=1000) -> bool:
        """Check if the number of tokens in each chunk is less than max_tokens,
            using tiktoken
        """
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        for doc in documents:
            chunks = doc.chunks
            for chunk in chunks:
                tokens = encoder.encode(chunk.text, disallowed_special=())
                chunk.set_tokens(tokens)
                if len(tokens) > max_tokens:
                    msg.fail(f"Chunk {chunk.id} has {len(tokens)} tokens, exceeding the limit of {max_tokens}")
                    return False
        return True


class EmbedderManager:
    def __init__(self) -> None:
        self.embedders: dict[str, Embedder] = {
            # TODO
        }
        self.selected_embedder: str = None

    def embed(self, documents: list[Document]) -> bool:
        msg.info(f"Starting embedding with {self.selected_embedder}")
        start_time = time.time()

        embedder = self.get_selected_embedder()
        success = embedder.embed(documents)
        msg.info(f"Embedding {'completed' if success else 'failed'} with {len(documents)} Documents and {sum([len(doc.chunks) for doc in documents])} Chunks in {time.time() - start_time:.2f} seconds")
        return success
    
    def set_embedder(self, embedder_name: str) -> bool:
        if embedder_name in self.embedders:
            msg.info(f"Setting embedder to {embedder_name}")
            self.selected_embedder = embedder_name
            return True
        else:
            msg.warn(f"Embedder {embedder_name} not found. Setting embedder to None")
            self.selected_embedder = None
            return False
    
    def get_selected_embedder(self) -> Embedder:
        return self.embedders.get(self.selected_embedder, None)
    

class RetrieverManager:
    def __init__(self) -> None:
        self.retrievers: dict[str, Retriever] = {
            "kendra": KendraRetriever(),
            # "faiss": FAISSRetriever(),
            "knowledge-base": KnowledgeBaseRetriever(),
        }
        self.selected_retriever: str = "knowledge-base"
        self.top_k = 5

    def retrieve(self, 
        queries: list[str], 
        embedder: Embedder, 
        top_k: int | None = None,
    ) -> dict[str, dict[str, list[Chunk]]]:
        msg.info(f"Starting retrieval with {self.selected_retriever}")
        start_time = time.time()
       
        base_top_k, addtional_top_k = utils.managed_top_k(self.top_k if top_k is None else top_k)

        retriever = self.get_selected_retriever()
        chunks_accumulated = {}
        for query in queries:
            chunks_accumulated[query] = {"base": [], "additional": []}
            msg.info(f"Retrieving with query: {query}")

            # base context retrieval
            msg.info("Retrieving base context...")
            retrieved_base_chunks = retriever.retrieve(query, embedder, top_k=base_top_k, category="base")
            chunks_accumulated[query]["base"] += retrieved_base_chunks
            msg.good(f"Retrieved {len(retrieved_base_chunks)} base chunks")

            # additional context retrieval
            base_doc_ids = set([chunk.doc_id for chunk in retrieved_base_chunks])
            msg.info(f"Retrieving additional context from {len(base_doc_ids)} base documents...")
            for base_doc_id in base_doc_ids:
                msg.info(f"Retrieving additional context for base doc id: {base_doc_id}...")
                retrieved_additional_chunks = retriever.retrieve(query, embedder, top_k=addtional_top_k, category="additional", base_doc_id=base_doc_id)
                chunks_accumulated[query]["additional"] += retrieved_additional_chunks
                msg.good(f"Retrieved {len(retrieved_additional_chunks)} additional chunks")

        total_base_chunks = sum([len(chunks_accumulated[query]["base"]) for query in queries])
        total_additional_chunks = sum([len(chunks_accumulated[query]["additional"]) for query in queries])
        msg.good(f"Retrieval completed with {total_base_chunks + total_additional_chunks} Chunks in {time.time() - start_time:.2f} seconds")
        return chunks_accumulated

    def rerank(self, chunks: dict[str, dict[str, list[Chunk]]], top_k: int=None, k=60) -> dict[str, list[Chunk]]:
        # {"<query>": {"base": list[Chunk], "additional": list[Chunk], ...}
        base_top_k, additional_top_k = utils.managed_top_k(self.top_k if top_k is None else top_k)
        
        total_base_chunks = sum([len(chunks[query]["base"]) for query in chunks])
        total_additional_chunks = sum([len(chunks[query]["additional"]) for query in chunks])
        msg.info(f"Reranking {total_base_chunks} base chunks and {total_additional_chunks} additional chunks")

        flattened_chunks = {}
        for query in chunks:
            for doc_type in chunks[query]:
                for chunk in chunks[query][doc_type]:
                    flattened_chunks[chunk.chunk_id] = chunk

        # reciprocal rank fusion for base chunks
        fused_scores = {}
        for query in chunks:
            base_chunks = chunks[query]["base"]
            base_chunks = sorted(base_chunks, key=lambda x: float(x.score), reverse=True)   

            for rank, chunk in enumerate(base_chunks):
                if chunk.chunk_id not in fused_scores:
                    fused_scores[chunk.chunk_id] = 0
                fused_scores[chunk.chunk_id] += 1 / (rank + k)
            
        reranked_base_chunks = []
        for chunk_id in fused_scores:
            reranked_base_chunks.append((flattened_chunks[chunk_id], fused_scores[chunk_id]))
        # TODO introduce tie-breaker: recency. Currently, original score is used as tie-breaker
        reranked_base_chunks: list[Chunk] = list(map(lambda x: x[0], sorted(reranked_base_chunks, key=lambda x: (float(x[1]), float(x[0].score)), reverse=True)))[:base_top_k]
        reranked_base_chunks_doc_ids = set([chunk.doc_id for chunk in reranked_base_chunks])

        # reciprocal rank fusion for additional chunks
        fused_scores = {}
        for query in chunks:
            additional_chunks = chunks[query]["additional"]
            additional_chunks = sorted(additional_chunks, key=lambda x: float(x.score), reverse=True)

            for rank, chunk in enumerate(additional_chunks):
                # skip if base doc is not in base chunks
                if chunk.chunk_meta.get("base_doc_id", "") not in reranked_base_chunks_doc_ids:
                    continue

                if chunk.chunk_id not in fused_scores:
                    fused_scores[chunk.chunk_id] = 0
                fused_scores[chunk.chunk_id] += 1 / (rank + k)

        reranked_additional_chunks = []
        for chunk_id in fused_scores:
            reranked_additional_chunks.append((flattened_chunks[chunk_id], fused_scores[chunk_id]))
        reranked_additional_chunks: list[Chunk] = list(map(lambda x: x[0], sorted(reranked_additional_chunks, key=lambda x: float(x[1]), reverse=True)))[:additional_top_k]

        msg.good(f"Reranking completed with {len(reranked_base_chunks)} base chunks and {len(reranked_additional_chunks)} additional chunks")
        return {"base": reranked_base_chunks, "additional": reranked_additional_chunks}

    def cutoff_text(self, text: str, content_length: int) -> str:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        encoded_tokens = encoding.encode(text, disallowed_special=())

        if len(encoded_tokens) > content_length:
            truncated_tokens = encoded_tokens[:content_length]
            truncated_text = encoding.decode(truncated_tokens)
            msg.info(f"Truncated text from {len(encoded_tokens)} tokens to {len(truncated_tokens)} tokens")
            return truncated_text
        else:
            msg.info(f"Text has {len(encoded_tokens)} tokens, not truncating")
            return text
        
    def sort_combined_chunks(self, docs: dict[str, CombinedChunks]) -> dict[str, CombinedChunks]:
        for doc_id in docs:
            docs[doc_id].chunks = dict(sorted(docs[doc_id].chunks.items(), key=lambda x: float(x[1].score), reverse=True))
        docs = dict(sorted(docs.items(), key=lambda x: x[1].score, reverse=True))
        return docs

    def combine_chunks(self, chunks: list[Chunk], doc_type: str="base", attach_url: bool=False) -> dict[str, CombinedChunks]:
        docs: dict[str, CombinedChunks] = {}
        for chunk in chunks:
            if chunk.doc_id not in docs:
                docs[chunk.doc_id] = CombinedChunks(doc_id=chunk.doc_id, doc_name=chunk.doc_meta.get("doc_name", ""), doc_type=doc_type)
                if doc_type == "additional":
                    docs[chunk.doc_id].doc_name = chunk.doc_meta.get("doc_name", "")
                if attach_url:
                    docs[chunk.doc_id].url = utils.get_presigned_url(chunk.doc_id)
            docs[chunk.doc_id].score += float(chunk.score)
            docs[chunk.doc_id].chunks[chunk.chunk_id] = chunk

        for doc_id in docs:
            chunk_cnt = len(docs[doc_id].chunks)
            docs[doc_id].num_chunks = chunk_cnt
        
        return docs

    def combine_context(self, base_docs: dict[str, CombinedChunks], additional_docs: dict[str, CombinedChunks]) -> dict[str, str]:
        context = {
            "base": "",
            "additional": "",
        }
        
        # base context
        for doc_id in base_docs:
            context["base"] += f"--- Document: {base_docs[doc_id].doc_name} ---\n\n"

            for chunk in base_docs[doc_id].chunks.values():
                context["base"] += f"{chunk.to_detailed_str()}\n\n"
        
        # additional context
        for doc_id in additional_docs:
            context["additional"] += f"--- Document: {additional_docs[doc_id].doc_name} ---\n\n"

            for chunk in additional_docs[doc_id].chunks.values():
                context["additional"] += f"{chunk.to_detailed_str()}\n\n"

        return context

    def set_retriever(self, retriever_name: str) -> bool:
        if retriever_name in self.retrievers:
            msg.info(f"Setting retriever to {retriever_name}")
            self.selected_retriever = retriever_name
            return True
        else:
            msg.warn(f"Retriever {retriever_name} not found. Setting retriever to None")
            self.selected_retriever = None
            return False

    def get_selected_retriever(self) -> Retriever:
        return self.retrievers.get(self.selected_retriever, None)

class GeneratorManager:
    def __init__(self) -> None:
        self.generators: dict[str, Generator] = {
            "gpt4": GPT4Generator(),
            "gpt3": GPT3Generator(),
        }
        self.selected_generator: str = "gpt4"
    
    def generate(self, queries: list[str], contexts: list[str], history: dict=None) -> str:
        msg.info(f"Starting generation with {self.selected_generator}")
        start_time = time.time()

        generator = self.get_selected_generator()

        if not history:
            history = {}
        truncated_history = self.truncate_history(history, max_tokens=generator.context_window * 0.3) # TODO

        response = generator.generate(queries, contexts, truncated_history)
        msg.info(f"Generation completed in {time.time() - start_time:.2f} seconds")
        return response
    
    def generate_stream(self, queries: list[str], context: dict[str, str], history: list[dict]=None):
        msg.info(f"Starting stream generation with {self.selected_generator}")

        generator = self.get_selected_generator()
        if not history:
            history = {}
        truncated_history = utils.truncate_history(history, max_tokens=generator.context_window * 0.3)

        for response in generator.generate_stream(queries, context, truncated_history):
            yield response

    def set_generator(self, generator_name: str) -> bool:
        if generator_name in self.generators:
            msg.info(f"Setting generator to {generator_name}")
            self.selected_generator = generator_name
            return True
        else:
            msg.warn(f"Generator {generator_name} not found. Setting generator to None")
            self.selected_generator = None
            return False
    
    def get_selected_generator(self) -> Generator:
        return self.generators.get(self.selected_generator, None)
    

class RevisorManager: # TODO integrate with GeneratorManager
    def __init__(self) -> None:
        self.revisor: dict[str, Revisor] = {
            "gpt": GPTRevisor(),
        }
        self.selected_revisor: str = "gpt"
    
    def revise(self, queries: list[str], history: dict=None, revise_prompt: ChatPromptTemplate=None) -> str:
        msg.info(f"Starting query transformation with {self.selected_revisor}")
        start_time = time.time()

        revisor = self.get_selected_revisor()

        if not history:
            history = {}
        truncated_history = utils.truncate_history(history, max_tokens=revisor.context_window * 0.7) # TODO

        response = revisor.revise(queries, truncated_history, revise_prompt)
        msg.good(f"Transformation completed in {time.time() - start_time:.2f} seconds")
        return response

    def set_revisor(self, revisor_name: str) -> bool:
        if revisor_name in self.revisor:
            msg.info(f"Setting revisor to {revisor_name}")
            self.selected_revisor = revisor_name
            return True
        else:
            msg.warn(f"Revisor {revisor_name} not found. Setting revisor to None")
            self.selected_revisor = None
            return False
    
    def get_selected_revisor(self) -> Revisor:
        return self.revisor.get(self.selected_revisor, None)


class ReaderManager:
    def __init__(self) -> None:
        self.readers: dict[str, Reader] = {
            # TODO
        }
        self.selected_reader: str = None

    def load(self, fileData: list[FileData]) -> list[Document]:
        # TODO
        return

class RAGManager:
    def __init__(self) -> None:
        self.reader_manager = ReaderManager()
        self.chunker_manager = ChunkerManager()
        self.embedder_manager = EmbedderManager()
        self.retriever_manager = RetrieverManager()
        self.generator_manager = GeneratorManager()
        self.revisor_manager = RevisorManager()
        self.config = {
            "model": {},
            "pipeline": {},
        }

    def set_config(self, config: dict) -> None:
        self.config = {
            "model": config.get("model", {}),
            "pipeline": config.get("pipeline", {}),
        }
        model_config = self.config["model"]

        self.reader_manager.selected_reader = model_config.get("Reader", {}).get("selected", None)
        self.chunker_manager.selected_chunker = model_config.get("Chunker", {}).get("selected", None)
        self.embedder_manager.selected_embedder = model_config.get("Embedder", {}).get("selected", None)
        self.retriever_manager.selected_retriever = model_config.get("Retriever", {}).get("selected", None)
        self.generator_manager.selected_generator = model_config.get("Generator", {}).get("selected", None)
        self.revisor_manager.selected_revisor = model_config.get("Revisor", {}).get("selected", None)

        # log manager configs
        msg.info(f"Setting READER to {self.reader_manager.selected_reader}")
        msg.info(f"Setting CHUNKER to {self.chunker_manager.selected_chunker}")
        msg.info(f"Setting EMBEDDER to {self.embedder_manager.selected_embedder}")
        msg.info(f"Setting RETRIEVER to {self.retriever_manager.selected_retriever}")
        msg.info(f"Setting GENERATOR to {self.generator_manager.selected_generator}")
        msg.info(f"Setting REVISOR to {self.revisor_manager.selected_revisor}")

    def index_documents(self, files: list[FileData]) -> list[Document]:
        pass

    def retrieve_chunks(self, queries: list[str], history: dict=None, top_k: int=None,
                        revise_query: bool=None, hyde: bool=None) -> tuple[list[Chunk], dict[str, str]]:
        # if revise_query and hyde are provided, they get priority over the config

        # pre-retrieval
        q = []
        if (isinstance(revise_query, bool) and revise_query) or self.config["pipeline"].get("revise_query", False):
            _history = {} if history is None else history
            # Rewrite query
            msg.info("Rewriting queries...")
            rewrited_query = self.revisor_manager.revise(queries, _history)
            # msg.info(f"Revised queries: {rewrited_query}")
            q.append(rewrited_query)
        if (isinstance(hyde, bool) and hyde) or self.config["pipeline"].get("hyde", False):
            # HyDE
            msg.info("Applying HyDE...")
            hyde_answer = self.revisor_manager.revise(queries, _history, revise_prompt=hyde_prompt)
            # msg.info(f"HyDE answer: {hyde_answer}")
            q.append(hyde_answer)

        _queries = q if q else queries
        # TODO Multi-RAG
        chunks = self.retriever_manager.retrieve(
                                queries=_queries,
                                embedder=self.embedder_manager.get_selected_embedder(),
                                top_k=top_k,
                          )
        
        # post-retrieval
        chunks = self.retriever_manager.rerank(chunks, top_k=top_k)
        base_chunks, additional_chunks = chunks.get("base", []), chunks.get("additional", [])
        base_docs = self.retriever_manager.combine_chunks(base_chunks)
        additional_docs = self.retriever_manager.combine_chunks(additional_chunks, doc_type="additional")

        context = self.retriever_manager.combine_context(base_docs, additional_docs)
        generator_context_window = self.generator_manager.get_selected_generator().context_window
        managed_context = {
            "base": self.retriever_manager.cutoff_text(context["base"], generator_context_window),
            "additional": self.retriever_manager.cutoff_text(context["additional"], generator_context_window), # TODO sum of length of contexts
        }
        
        return base_chunks, additional_chunks, managed_context

    def retrieve_document(self, doc_id: str) -> Document:
        pass

    def generate_answer(self, queries: list[str], context: list[str]) -> str:
        pass

    def generate_stream_answer(self, queries: list[str], context: dict[str, str], history: list[dict]=None):
        for response in self.generator_manager.generate_stream(queries, context, history):
            yield response

    def reset(self) -> None:
        pass

    def combine_chunks(self, chunks: list[Chunk], doc_type: str, attach_url=False) -> dict[str, CombinedChunks]:
        res = self.retriever_manager.combine_chunks(chunks, doc_type=doc_type, attach_url=attach_url)
        res = self.retriever_manager.sort_combined_chunks(res)
        return res

