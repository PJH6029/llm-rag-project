from wasabi import msg
import base64, io
from datetime import datetime

from rag.interfaces import Reader
from rag.types import Document, FileData

from pypdf import PdfReader

class BasicReader(Reader):
    def __init__(self) -> None:
        super().__init__()
        self.name = "BasicReader"

    def load(self, fileData: list[FileData]) -> list[Document]:
        documents = []

        for file in fileData:
            msg.info(f"Loading in {file.filename}")
            
            decoded_bytes = base64.b64decode(file.content)

            if file.extension == "pdf":
                try:
                    pdf_bytes = io.BytesIO(base64.b64decode(file.content))

                    full_text = ""
                    reader = PdfReader(pdf_bytes)

                    for page in reader.pages:
                        full_text += page.extract_text() + "\n\n"
                    
                    document = Document(
                        text=full_text,
                        type="Document",
                        timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        reader=self.name,
                    )
                    # TODO add metadata

                    documents.append(document)
                except Exception as e:
                    msg.warn(f"Failed to load {file.filename}: {e}")
            else:
                # TODO
                msg.warn(
                    f"{file.filename} with extension {file.extension} not supported by BasicReader. Skipping..."
                )
        return documents