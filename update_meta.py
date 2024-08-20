from pinecone import Pinecone
import os

from rag.component.vectorstore.PineconeVectorstore import PineconeVectorstore
from rag.component import llm, embeddings

META_KEY_PREFIX = "doc_meta___"

doc_ids = [
    # "APPLE/ISO_SSD_NVME_FADU_FIRMWARE_NOMENCLATURE_1.3-DRAFT.pdf",
    # "APPLE/ISO_SSD_NVME_GENERIC_MANUFACTURING_ATTRIBUTES_1.0.pdf",
    # "APPLE/ISO_SSD_PCIE_NVME_ERS_GENERIC_v1.2a.pdf",
    "MS/Microsoft_v3.20_CloudSSD_specification_correction marks.pdf",
]
prefix = f"s3://{os.environ['S3_BUCKET_NAME']}/frequently_access_documents"
doc_ids = [f"{prefix}/{doc_id}" for doc_id in doc_ids]

namespaces = [
    "parent-upstage-overlap-backup",
    "child-upstage-overlap-backup",
]

api_key = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=api_key)
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

# embeddings_ = embeddings.get_model("text-embedding-3-small")
# for doc_id in doc_ids:
#     print(doc_id)
#     for namespace in namespaces:
#         vectorstore = PineconeVectorstore(embeddings=embeddings_, namespace=namespace)

#         chunks = vectorstore.query("", top_k=505, filter={"doc_id": doc_id})

#         chunk_ids = [chunk.chunk_id for chunk in chunks]
        
#         for chunk_id in chunk_ids:
#             index.update(
#                 id=chunk_id,
#                 set_metadata={f"{META_KEY_PREFIX}_category": "additional",f"{META_KEY_PREFIX}base-doc-id": "*"},
#                 namespace=namespace,
#             )
#         print(f"Updated {len(chunk_ids)} chunks in {namespace}")

doc_id_prefix = "s3://llm-project-demo-bucket/frequently_access_documents/"
total = 0
for ids in index.list(namespace="child-upstage-overlap-backup"):
    fetched = index.fetch(
        ids=ids,
        namespace="child-upstage-overlap-backup",
    )
    for record in fetched["vectors"].values():
        metadata = record["metadata"]
        doc_id = metadata["doc_id"]
        doc_type = "/".join(doc_id.split('/')[4:-1])
        chunk_id = record["id"]
        index.update(
            id=chunk_id,
            set_metadata={f"{META_KEY_PREFIX}doc_type": doc_type},
            namespace="child-upstage-overlap-backup",
        )
    print(f"Updated {len(fetched['vectors'])} chunks in child-upstage-overlap-backup")
    total += len(fetched['vectors'])
print(f"Total updated chunks: {total}")