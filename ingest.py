import os, json

from rag.api import upload_data, ingest_data
from rag.component.loader import UpstageLayoutBackupDirLoader, PDFWithMetadataLoader
from rag.component.ingestor.PineconeMultiVectorIngestor import PineconeMultiVectorIngestor

# bucket = os.environ["S3_BUCKET_NAME"]
# files = [
#     # "OCP/2.0/Datacenter NVMe SSD Specification v2.0r21.pdf", d
    
#     # "OCP/2.0/Datacenter_NVMe_SSD_Specification_v2.0_Addendum_v0.6_changebar.pdf", d
#     # "OCP/2.0/Datacenter_NVMe_SSD_Specification_v2.0_Addendum_v0.7.pdf", d
#     # "OCP/2.0/Datacenter_NVMe_SSD_Specification_v2.0_Addendum_v0.8[82].pdf", d
#     # "OCP/2.0/Datacenter_NVMe_SSD_Specification_v2.0_Addendum_v0.91.pdf", d
    
#     # "OCP/2.5/Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf", d
#     # "OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf", d
    
#     # "OCP/Boot Drive/HyperScale-NVMe-Boot-SSD-Specification-v1.0.pdf", d
#     # "OCP/Boot Drive/NVMe-Cloud-SSD-Specification_v1.0_Addendum_v0.8.pdf", d
    
#     # "NVMe/NVMe-NVM-Express-2.0a-2021.07.26-Ratified.pdf", d
#     # "NVMe/NVM-Express-Management-Interface-Specification-1.2b-2022.01.10-Ratified.pdf", d
    
#     # "PCIe/NCB-PCI_Express_Base_r3.1a_December7-2015.pdf", d
#     # "PCIe/PCIE_Base_Specification_Revision_4_0_Version 1_0.pdf", d
#     # "PCIe/PCI Express_ Base Specification Revision 5.0 Version 1.0.pdf", d
#     # "PCIe/NCB-PCI_Express_Base_6.2-2024-01-25.pdf", d
    
#     "APPLE/ISO_SSD_NVME_FADU_FIRMWARE_NOMENCLATURE_1.3-DRAFT.pdf",
#     "APPLE/ISO_SSD_NVME_GENERIC_MANUFACTURING_ATTRIBUTES_1.0.pdf",
#     "APPLE/ISO_SSD_PCIE_NVME_ERS_GENERIC_v1.2a.pdf",
    
#     "MS/Microsoft_v3.20_CloudSSD_specification_correction marks.pdf",
    
#     "MCTP/DSP0236_1.3.1(MCTP_Base_Spec).pdf",
    
#     "NVMe-MI/NVM-Express-Management-Interface-1.1d-2021.04.19-Ratified.pdf",
#     "NVMe-MI/NVM-Express-Management-Interface-Specification-1.2b-2022.01.10-Ratified.pdf",
#     "NVMe-MI/NVM-Express-Management-Interface-Specification-1.2c-2022.10.06-Ratified.pdf",
# ]

# def upload():
#     prefix = os.path.join(os.path.dirname(__file__), "ref_docs", "Frequently Access Documents")
    
#     for file in files:
#         filename = os.path.basename(file)
#         location = os.path.dirname(file)
#         upload_data(os.path.join(prefix, file), f"frequently_access_documents/{location}")

# def ingest():
#     cnt = 0
#     PineconeMultiVectorIngestor.CHILD_INGESTION_CNT = 0
#     for file in files:
#         s3_url = f"s3://{bucket}/frequently_access_documents/{file}"
#         cnt += ingest_data(s3_url)
    
#     print(f"{cnt} parent chunks ingested")
#     print(f"{PineconeMultiVectorIngestor.CHILD_INGESTION_CNT} child chunks ingested")

# if __name__ == "__main__":
#     upload()
#     # ingest()



backup_dir = os.path.join(os.path.dirname(__file__), "layout_overlap_backup")
ref_docs_root = os.path.join(os.path.dirname(__file__), "ref_docs", "Frequently Access Documents")

key_map = {
    "_category": "doc_type",
    "version": "version",
    "base-doc-id": "base_doc_id",
}

# bucket = os.environ["S3_BUCKET_NAME"]
# for root, _, files in os.walk(ref_docs_root):
#     for file in files:
#         if not file.endswith(".pdf"):
#             continue
#         file_path = os.path.join(root, file)
#         metadata_path = file_path + ".metadata.json"
        
#         with open(metadata_path, "r") as f:
#             _metadata_dict = json.load(f)
#         _metadata_dict = _metadata_dict["Attributes"]
        
#         metadata_dict = {
#             key_map.get(k, k): v
#             for k, v in _metadata_dict.items()
#         }
        
#         category = "/".join(file_path.split("/")[7:-1])
#         metadata_dict["category"] = category
        
#         object_key = f"frequently_access_documents/{category}/{file}"
#         s3_url = f"s3://{bucket}/{object_key}"
#         metadata_dict["doc_id"] = s3_url
        
#         file_name_without_ext = os.path.splitext(file)[0]
#         target_parent_dirs = [
#             f"{backup_dir}/html",
#             f"{backup_dir}/markdown"
#         ]
#         for target_parent_dir in target_parent_dirs:
#             target_dir = os.path.join(target_parent_dir, file_name_without_ext)
            
#             if not os.path.exists(target_dir):
#                 continue
            
#             # list files in target dir
#             for target_file in os.listdir(target_dir):
#                 if not (target_file.endswith(".html") or target_file.endswith(".md")):
#                     continue
                
#                 target_file_name_without_ext = os.path.splitext(target_file)[0]
#                 page = int(target_file_name_without_ext.split("_")[-1])
#                 metadata_dict["page"] = page
                
#                 metadata_path = os.path.join(target_dir, target_file + ".metadata.json")
#                 # write metadata to metadata file
#                 with open(metadata_path, "w") as f:
#                     json.dump(metadata_dict, f)
#                 print(f"Metadata written to {metadata_path}")
        
# def metadata_handler(metadata: dict) -> tuple[dict, dict]:
#     doc_meta = {
#         "doc_id": metadata["doc_id"],
#         "doc_type": metadata["doc_type"],
#         "version": metadata["version"],
#         "category": metadata["category"],
#     }
#     if metadata.get("base_doc_id"):
#         doc_meta["base_doc_id"] = metadata["base_doc_id"]
    
#     chunk_meta = {
#         "page": metadata["page"],
#     }

PineconeMultiVectorIngestor.CHILD_INGESTION_CNT = 0
PineconeMultiVectorIngestor.INGEST_FROM_SCRATCH = False
cnt = ingest_data(
    loader=UpstageLayoutBackupDirLoader(backup_dir),
    batch_size=20,
)
print(f"{cnt} parent chunks ingested")
print(f"{PineconeMultiVectorIngestor.CHILD_INGESTION_CNT} child chunks ingested")
