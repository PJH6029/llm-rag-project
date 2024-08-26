import os, json, argparse
from pathlib import Path
from typing import Type, Callable, Any

from langchain_community.document_loaders import PyPDFLoader

from rag.component.loader import *
from rag.api import upload_data, ingest_data, get_config
from rag.component.ingestor import PineconeMultiVectorIngestor
from rag import util

parser = argparse.ArgumentParser(description="Ingest data")
parser.add_argument(
    "-l",
    "--loader",
    type=str,
    metavar="",
    required=False,
    help="Loader name. e.g. upstage_layout, upstage_backup, pypdf. See rag/managers/loader.py. Default: pypdf",
    default="pypdf",
)
parser.add_argument(
    "-s",
    "--source_dir",
    type=str,
    metavar="",
    required=False,
    help="Source directory. Default: source_documents",
    default="source_documents",
)
parser.add_argument(
    "-b",
    "--backup_dir",
    type=str,
    metavar="",
    required=False,
    help="Backup directory. Used for caching layout parse result. Default: backup",
    default="backup",
)
parser.add_argument(
    "-a",
    "--all",
    action="store_true",
    help="Ingest all files in the source directory. If not set, scan ingesting log and ingest only missing files.",
)
parser.add_argument(
    "-d",
    "--download",
    action="store_true",
    help="Download all files from S3 to the source directory. The name of the source directory is defined by -s option.",
)
args = parser.parse_args()

backup_dir = os.path.join(os.path.dirname(__file__), args.backup_dir)
source_dir = os.path.join(os.path.dirname(__file__), args.source_dir)

source_exts = [".pdf",]

persistent_metadata_handler = lambda metadata: util.persistent_metadata_handler(metadata, source_dir=source_dir)

def _ingest(loader: BaseLoader):
    cnt = ingest_data(
        loader=loader,
        batch_size=10,
    )
    print(f"{cnt} chunks ingested")
    
def _ingest_walk(loader_init: Callable[[Any], BaseLoader], **kwargs):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if any(file.endswith(ext) for ext in source_exts):
                _ingest(loader_init(
                    os.path.join(root, file),
                    **kwargs,
                ))

def ingest():
    print(f"Using loader: {args.loader}")
    if args.loader == "upstage_layout":
        # ingest all pdf files in the source_dir
        _ingest_walk(
            UpstageLayoutLoader,
            overlap_elem_size=2,
            use_ocr=True,
            to_markdown=True,
            cache_to_local=True,
            backup_dir=backup_dir,
            metadata_handler=persistent_metadata_handler,
        )
    elif args.loader == "upstage_backup":
        # ingest all backup files in the backup_dir
        _ingest(UpstageLayoutBackupDirLoader(
            backup_dir,
            metadata_handler=persistent_metadata_handler,
        ))
    elif args.loader == "pypdf":
        def loader_init(file_path: str, **kwargs):
            return BaseRAGLoader.from_lc_loader(
                PyPDFLoader(
                    file_path,
                    **kwargs,
                ),
                metadata_handler=persistent_metadata_handler,
            )
        
        _ingest_walk(
            loader_init,
        )
    else:
        raise ValueError(f"Unknown loader: {args.loader}")

def get_file_folders(s3_client, bucket_name, prefix=""):
    file_names = []
    folders = []

    default_kwargs = {
        "Bucket": bucket_name,
        "Prefix": prefix
    }
    next_token = ""

    while next_token is not None:
        updated_kwargs = default_kwargs.copy()
        if next_token != "":
            updated_kwargs["ContinuationToken"] = next_token

        response = s3_client.list_objects_v2(**updated_kwargs)
        contents = response.get("Contents")

        for result in contents:
            key = result.get("Key")
            if key[-1] == "/":
                folders.append(key)
            else:
                file_names.append(key)

        next_token = response.get("NextContinuationToken")

    return file_names, folders

def download_files(s3_client, bucket_name, local_path, file_names, folders, ignore_existing=False):
    local_path = Path(local_path)

    for folder in folders:
        folder_path = Path.joinpath(local_path, folder)
		# Create all folders in the path
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        file_path = Path.joinpath(local_path, file_name)
		# Create folder for parent directory
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if ignore_existing and file_path.exists():
            print(f"File {file_path} already exists. Skipping...")
            continue
        
        print(f"Downloading {file_name} to {file_path}")
        s3_client.download_file(
            bucket_name,
            file_name,
            str(file_path)
        )

def attach_metadata(local_path, file_names, bucket_name, metadata_ext=".metadata.json"):
    for file_name in file_names:
        if not file_name.endswith(".pdf"):
            continue
        metadata_path = os.path.join(local_path, file_name) + metadata_ext
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        metadata["source"] = f"s3://{bucket_name}/{file_name}"
        metadata["doc_id"] = metadata["source"]
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)


def download_from_s3(to_dir: str, ignore_existing=True):
    import boto3
    s3_bucket = os.getenv("S3_BUCKET_NAME")
    s3_client = boto3.client("s3")
    file_names, folders = get_file_folders(s3_client, s3_bucket)
    download_files(s3_client, s3_bucket, to_dir, file_names, folders, ignore_existing=ignore_existing)
    attach_metadata(to_dir, file_names, s3_bucket)


def main():
    if args.download:
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
        download_from_s3(to_dir=source_dir, ignore_existing=not args.all)
        
    print(f"Source directory: {source_dir}")
    print(f"Backup directory: {backup_dir}")
    print(f"Pinecone Index: {os.getenv('PINECONE_INDEX_NAME')}")
    
    if get_config().ingestion.ingestor == "pinecone-multivector":
        PineconeMultiVectorIngestor.CHILD_INGESTION_CNT = 0
        PineconeMultiVectorIngestor.INGEST_FROM_SCRATCH = args.all
    ingest()
    if get_config().ingestion.ingestor == "pinecone-multivector":
        print(f"{PineconeMultiVectorIngestor.CHILD_INGESTION_CNT} child chunks ingested")

if __name__ == "__main__":
    main()