class Chunk:
    def __init__(
        self,
        text: str="",
        doc_name: str="",
        doc_type: str="",
        doc_id: str="",
        chunk_id: str="",
        meta: dict=None,
    ):
        self._text = text
        self._doc_name = doc_name
        self._doc_type = doc_type
        self._doc_id = doc_id
        self._chunk_id = chunk_id
        self._tokens = 0
        self._score = 0.0

        if meta is None:
            meta = {}
        self._meta = meta
    
    @property
    def text(self):
        return self._text

    @property
    def doc_name(self):
        return self._doc_name

    @property
    def doc_type(self):
        return self._doc_type

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def chunk_id(self):
        return self._chunk_id

    @property
    def tokens(self):
        return self._tokens

    @property
    def score(self):
        return self._score
    
    @property
    def meta(self):
        return self._meta
    
    @doc_id.setter
    def doc_id(self, id):
        self._doc_id = id
    
    @tokens.setter
    def tokens(self, tokens):
        self._tokens = tokens
    
    @score.setter
    def score(self, score):
        self._score = score

    @meta.setter
    def meta(self, meta):
        self._meta = meta
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "doc_name": self.doc_name,
            "doc_type": self.doc_type,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "tokens": self.tokens,
            "score": self.score,
            "meta": self.meta,
        }
    
    def to_detailed_str(self):
        return (
            f"CHUNK ({self.chunk_id})\n"
            # f"from DOCUMENT [{self.doc_name}] (type: {self.doc_type}, id: {self.doc_id})\n" # alerady mentioned when combining context
            f"(\n"
            f"SIMILARITY_SCORE= {self.score}\n"
            f"TEXT=\n"
            f"{self.text}\n"
            f"METADATA=\n"
            f"{self.meta}\n"
            f")"
        )

    @classmethod
    def from_dict(data: dict):
        chunk = Chunk(
            text=data.get("text", ""),
            doc_name=data.get("doc_name", ""),
            doc_type=data.get("doc_type", ""),
            doc_id=data.get("doc_id", ""),
            chunk_id=data.get("chunk_id", ""),
            meta=data.get("meta", {}),
        )
        chunk.tokens = data.get("tokens", 0)
        chunk.score = data.get("score", 0.0)
        return chunk
    
class Document:
    def __init__(
        self,
        text: str="",
        type: str="",
        name: str="",
        path: str="",
        link: str="",
        timestamp: str="",
        reader: str="",
        meta: dict=None,
    ):
        if meta is None:
            meta = {}
        
        self._text = text
        self._type = type
        self._name = name
        self._path = path
        self._link = link
        self._timestamp = timestamp
        self._reader = reader
        self._meta = meta
        self.chunks: list[Chunk] = []
    
    @property
    def text(self):
        return self._text
    
    @property
    def type(self):
        return self._type
    
    @property
    def name(self):
        return self._name
    
    @property
    def path(self):
        return self._path
    
    @property
    def link(self):
        return self._link
    
    @property
    def timestamp(self):
        return self._timestamp
    
    @property
    def reader(self):
        return self._reader
    
    @property
    def meta(self):
        return self._meta
    
    def to_dict(self) -> dict:
        doc_dict = {
            "text": self.text,
            "type": self.type,
            "name": self.name,
            "path": self.path,
            "link": self.link,
            "timestamp": self.timestamp,
            "reader": self.reader,
            "meta": self.meta,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        return doc_dict
    
    @staticmethod
    def from_dict(data: dict):
        document = Document(
            text=data.get("text", ""),
            type=data.get("type", ""),
            name=data.get("name", ""),
            path=data.get("path", ""),
            link=data.get("link", ""),
            timestamp=data.get("timestamp", ""),
            reader=data.get("reader", ""),
            meta=data.get("meta", {}),
        )
        document.chunks = [Chunk.from_dict(chunk) for chunk in data.get("chunks", [])]
        return document
    

class FileData:
    pass