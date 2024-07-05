from wasabi import msg
import time
import tiktoken

from rag.interfaces import *
from rag.types import *
from rag.generate.GPT4Generator import GPT4Generator
from rag.generate.GPT3Generator import GPT3Generator
from rag.retrieve.KendraRetriever import KendraRetriever

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
        }
        self.selected_retriever: str = "kendra"

    def retrieve(self, 
        queries: list[str], 
        embedder: Embedder, 
        generator: Generator,
        query_revisor: Generator=None,
    ) -> tuple[list[Chunk], str]:
        if query_revisor:
            queries = query_revisor.generate() # TODO
        
        msg.info(f"Starting retrieval with {self.selected_retriever}, queries: {queries}")
        start_time = time.time()

        retriever = self.get_selected_retriever()
        chunks, context = retriever.retrieve(queries, embedder)

        managed_context = retriever.cutoff_text(context, content_length=generator.context_window)
        msg.good(f"Retrieval completed with {len(chunks)} Chunks in {time.time() - start_time:.2f} seconds")
        return chunks, managed_context

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
    
    def generate_stream(self, queries: list[str], contexts: list[str], history: list[dict]=None):
        msg.info(f"Starting stream generation with {self.selected_generator}")

        generator = self.get_selected_generator()
        if not history:
            history = {}
        truncated_history = self.truncate_history(history, max_tokens=generator.context_window * 0.3)
        history_str = self.history_to_str(truncated_history)

        for response in generator.generate_stream(queries, contexts, history_str):
            yield response
    
    def truncate_history(self, history: list[dict], max_tokens: int) -> dict:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        sum_tokens = 0
        truncated_history = []

        # Start with newest history
        for item in reversed(history):
            tokens = encoding.encode(item["content"], disallowed_special=())

            if sum_tokens + len(tokens) > max_tokens:
                remaining_tokens = max_tokens - sum_tokens
                truncated_tokens = tokens[:remaining_tokens]
                truncated_content = encoding.decode(truncated_tokens)

                truncated_item = {
                    "role": item["role"],
                    "content": truncated_content,
                }
                truncated_history.append(truncated_item)
                break
            else:
                sum_tokens += len(tokens)
                truncated_history.append(item)
        return list(reversed(truncated_history))

    def history_to_str(self, history: list[dict]) -> str:
        return "\n".join([f"{item['role'].upper()}: {item['content']}" for item in history])

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

    def init(self, config: dict) -> None:
        pass

    def index_documents(self, files: list[FileData]) -> list[Document]:
        pass

    def retrieve_chunks(self, queries: list[str], revise_query=False) -> tuple[list[Chunk], str]:
        query_revisor=None
        if revise_query:
            # TODO
            pass

        chunks, context = self.retriever_manager.retrieve(
                                queries=queries,
                                embedder=self.embedder_manager.get_selected_embedder(),
                                generator=self.generator_manager.get_selected_generator(),
                                query_revisor=query_revisor
                          )
        return chunks, context

    def retrieve_document(self, doc_id: str) -> Document:
        pass

    def generate_answer(self, queries: list[str], context: list[str]) -> str:
        pass

    def generate_stream_answer(self, queries: list[str], context: list[str], history: list[dict]=None):
        for response in self.generator_manager.generate_stream(queries, context, history):
            yield response

    def reset(self) -> None:
        pass

    def combine_chunks(self, chunks: list[Chunk]) -> dict:
        return self.retriever_manager.get_selected_retriever().combine_chunks(chunks) # TODO proper implementation

