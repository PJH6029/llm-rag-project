import os, sys
from markdownify import markdownify as md


from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_upstage import UpstageLayoutAnalysisLoader

from rag.api import query, query_stream
from rag.util import *
from rag.component.vectorstore.PineconeVectorstore import PineconeVectorstore
from rag.component import embeddings

# if __name__ == "__main__":
#     query_ = "Power consumption measurement"
    
#     res = query(query_)
#     print(res)
    
    
# if __name__ == "__main__":
#     url = "./ref_docs/Frequently Access Documents/OCP/2.0/Datacenter_NVMe_SSD_Specification_v2.0_Addendum_v0.8[82].pdf"
#     # pypdfloader = PyPDFLoader(url)
#     upstageloader = UpstageLayoutAnalysisLoader(url)
    
#     # doc1 = pypdfloader.load()[:]
    
#     # doc2 = UpstageLayoutAnalysisLoader(url).load()
    
#     doc3 = UpstageLayoutAnalysisLoader(url, split="page", use_ocr=True).load()
    
#     # doc4 = UpstageLayoutAnalysisLoader(url, split="element").load()

#     # print("-" * 10)
    
#     for i, doc in enumerate(doc3):
#         print(f"---- DOC {i} ----")
#         print(doc.page_content)
#         print(doc.metadata)
#         markdown = md(doc.page_content)
#         print()
#         print(markdown)
#         print("\n\n")
        
#     print("-" * 10)

if __name__ == "__main__":
    embeddings_ = embeddings.get_model("text-embedding-3-small")
    vectorstore = PineconeVectorstore(
        embeddings=embeddings_,
        namespace="child-upstage-overlap-backup",
    )
    
    chunks = vectorstore.query(
        "Deallocate behavior",
    )
    
    print([chunk.doc_id for chunk in chunks])