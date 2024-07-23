import os, sys

from dotenv import load_dotenv
load_dotenv()

from rag.api import query, query_stream

if __name__ == "__main__":
    query_ = "Power consumption measurement"
    
    res = query(query_)
    print(res)