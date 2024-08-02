import os

from rag.api import upload_data, ingest_data
from rag.component.ingestor.PineconeMultiVectorIngestor import PineconeMultiVectorIngestor

bucket = os.environ["S3_BUCKET_NAME"]
files = [
    {
        "path": "Google/Google SSD Scheduling Requirements for Vendor v0.9 (TTD_2458)[1][2].pdf",
        "metadata": {
            "Attributes": {
                "_category": "additional",
                "version": "0.9",
                "base-doc-id": "*"
            }
        }
    },
]

def upload():
    prefix = os.path.join(os.path.dirname(__file__), "ref_docs", "Frequently Access Documents")
    
    for file in files:
        file_path, metadata = file["path"], file.get("metadata")
        filename = os.path.basename(file_path)
        location = os.path.dirname(file_path)
        upload_data(os.path.join(prefix, file_path), f"frequently_access_documents/{location}", metadata)

def ingest():
    cnt = 0
    PineconeMultiVectorIngestor.CHILD_INGESTION_CNT = 0
    for file in files:
        file_path = file["path"]
        s3_url = f"s3://{bucket}/frequently_access_documents/{file_path}"
        cnt += ingest_data(s3_url)
    
    print(f"{cnt} parent chunks ingested")
    print(f"{PineconeMultiVectorIngestor.CHILD_INGESTION_CNT} child chunks ingested")

if __name__ == "__main__":
    # upload()
    ingest()