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
        self.top_k = 6

    def retrieve(self, 
        queries: list[str], 
        embedder: Embedder, 
    ) -> dict[str, list[Chunk]]:
        msg.info(f"Starting retrieval with {self.selected_retriever}")
        start_time = time.time()

        retriever = self.get_selected_retriever()
        chunks_accumulated = { "base": [], "additional": []}
        for query in queries:
            msg.info(f"Retrieving with query: {query}")
            chunks = retriever.retrieve(query, embedder, top_k=self.top_k//2)
            for key in chunks:
                chunks_accumulated[key] += chunks[key]

        msg.good(f"Retrieval completed with {(len(chunks_accumulated['base']) + len(chunks_accumulated['additional']))} Chunks in {time.time() - start_time:.2f} seconds")
        return chunks_accumulated

    def rerank(self, chunks: dict[str, list[Chunk]]) -> tuple[dict, dict]:
        # doc_id, base_doc_id, score
        base_chunks = chunks.get("base", [])
        additional_chunks = chunks.get("additional", [])
        
        base_chunks = self.combine_chunks(base_chunks)
        additional_chunks = self.combine_chunks(additional_chunks, doc_type="additional")

        # reorder additional chunks based on base chunk ids
        # TODO
        # base_doc_ids = base_chunks.keys()

        return base_chunks, additional_chunks

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

    def combine_chunks(self, chunks: list[Chunk], doc_type: str="base") -> dict:
        docs = {}
        for chunk in chunks:
            if chunk.doc_id not in docs:
                docs[chunk.doc_id] = {"score": 0.0, "chunks": {}, "doc_name": chunk.doc_meta.get("doc_name", "")}
                if doc_type == "additional":
                    docs[chunk.doc_id]["base_doc_id"] = chunk.doc_meta.get("base_doc_id", "")
            docs[chunk.doc_id]["score"] += float(chunk.score)
            docs[chunk.doc_id]["chunks"][chunk.chunk_id] = chunk

        for doc_id in docs:
            chunk_cnt = len(docs[doc_id]["chunks"])
            docs[doc_id]["num_chunks"] = chunk_cnt
        
        # sort docs. preventing lost in the middle problem (see https://arxiv.org/abs/2307.03172)
        docs = dict(sorted(docs.items(), key=lambda x: x[1]["score"], reverse=True))
        return docs

    def combine_context(self, base_chunks: list[Chunk], additional_chunks: list[Chunk]) -> dict[str, str]:
        base_docs = self.combine_chunks(base_chunks)
        additional_docs = self.combine_chunks(additional_chunks, doc_type="additional")

        context = {
            "base": "",
            "additional": "",
        }
        
        # docs = dict(list(docs.items())[:self.top_k])

        # base context
        for doc_id in base_docs:
            # sort chunk by scores
            sorted_chunks: list[Chunk] = list(sorted(base_docs[doc_id]["chunks"].values(), key=lambda chunk: chunk.score, reverse=True))

            context["base"] += f"--- Document: {base_docs[doc_id]['doc_name']} ---\n\n"

            for chunk in sorted_chunks:
                context["base"] += f"{chunk.to_detailed_str()}\n\n"
        
        # additional context
        for doc_id in additional_docs:
            # sort chunk by scores
            sorted_chunks: list[Chunk] = list(sorted(additional_docs[doc_id]["chunks"].values(), key=lambda chunk: chunk.score, reverse=True))

            context["additional"] += f"--- Document: {additional_docs[doc_id]['doc_name']} ---\n\n"

            for chunk in sorted_chunks:
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
        msg.info(f"Starting query revision with {self.selected_revisor}")
        start_time = time.time()

        revisor = self.get_selected_revisor()

        if not history:
            history = {}
        truncated_history = utils.truncate_history(history, max_tokens=revisor.context_window * 0.7) # TODO

        response = revisor.revise(queries, truncated_history, revise_prompt)
        msg.good(f"Revision completed in {time.time() - start_time:.2f} seconds")
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

    def init(self, config: dict) -> None:
        pass

    def index_documents(self, files: list[FileData]) -> list[Document]:
        pass

    def retrieve_chunks(self, queries: list[str], history: dict=None, 
                        revise_query=False, hyde=False) -> tuple[list[Chunk], dict[str, str]]:
        # pre-retrieval
        q = []
        if revise_query:
            _history = {} if history is None else history
            # Rewrite query
            rewrited_query = self.revisor_manager.revise(queries, _history)
            msg.info(f"Revised queries: {rewrited_query}")
            q.append(rewrited_query)
        if hyde:
            # HyDE
            hyde_answer = self.revisor_manager.revise(queries, _history, revise_prompt=hyde_prompt)
            msg.info(f"HyDE answer: {hyde_answer}")
            q.append(hyde_answer)

        # TODO Multi-RAG
        chunks = self.retriever_manager.retrieve(
                                queries=q if q else queries,
                                embedder=self.embedder_manager.get_selected_embedder(),
                          )
        
        # post-retrieval
        base_chunks, additional_chunks = chunks.get("base", []), chunks.get("additional", [])
        # base_docs, additional_docs = self.rerank(chunks)
        context = self.retriever_manager.combine_context(base_chunks, additional_chunks)
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

    def combine_chunks(self, chunks: list[Chunk]) -> dict:
        return self.retriever_manager.combine_chunks(chunks)

