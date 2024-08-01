import os

from rag.api import upload_data, ingest_data
from rag.component.loader.PDFWithMetadataLoader import PDFWithMetadataLoader
from rag.component.loader.UpstageLayoutLoader import UpstageLayoutLoader
from rag.component.ingestor.PineconeMultiVectorIngestor import PineconeMultiVectorIngestor

bucket = os.environ["S3_BUCKET_NAME"]
files = [
    # "OCP/2.0/Datacenter NVMe SSD Specification v2.0r21.pdf", d
    
    # "OCP/2.0/Datacenter_NVMe_SSD_Specification_v2.0_Addendum_v0.6_changebar.pdf", d
    # "OCP/2.0/Datacenter_NVMe_SSD_Specification_v2.0_Addendum_v0.7.pdf", d
    # "OCP/2.0/Datacenter_NVMe_SSD_Specification_v2.0_Addendum_v0.8[82].pdf", d
    # "OCP/2.0/Datacenter_NVMe_SSD_Specification_v2.0_Addendum_v0.91.pdf", d
    
    # "OCP/2.5/Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf", d
    # "OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf", d
    
    # "OCP/Boot Drive/HyperScale-NVMe-Boot-SSD-Specification-v1.0.pdf", d
    # "OCP/Boot Drive/NVMe-Cloud-SSD-Specification_v1.0_Addendum_v0.8.pdf", d
    
    # "NVMe/NVMe-NVM-Express-2.0a-2021.07.26-Ratified.pdf", d
    # "NVMe/NVM-Express-Management-Interface-Specification-1.2b-2022.01.10-Ratified.pdf", d
    
    # "PCIe/NCB-PCI_Express_Base_r3.1a_December7-2015.pdf", d
    # "PCIe/PCIE_Base_Specification_Revision_4_0_Version 1_0.pdf", d
    # "PCIe/PCI Express_ Base Specification Revision 5.0 Version 1.0.pdf", d
    "PCIe/NCB-PCI_Express_Base_6.2-2024-01-25.pdf"
]

def upload():
    prefix = os.path.join(os.path.dirname(__file__), "ref_docs", "Frequently Access Documents")
    
    for file in files:
        filename = os.path.basename(file)
        location = os.path.dirname(file)
        upload_data(os.path.join(prefix, file), f"frequently_access_documents/{location}")

def ingest():
    cnt = 0
    PineconeMultiVectorIngestor.CHILD_INGESTION_CNT = 0
    for file in files:
        s3_url = f"s3://{bucket}/frequently_access_documents/{file}"
        cnt += ingest_data(s3_url)
    
    print(f"{cnt} parent chunks ingested")
    print(f"{PineconeMultiVectorIngestor.CHILD_INGESTION_CNT} child chunks ingested")

if __name__ == "__main__":
    # upload()
    ingest()