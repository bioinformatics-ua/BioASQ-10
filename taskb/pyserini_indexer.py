"""
Replication test on 21/07/2022 was sucessfull by Tiago Almeida

Command: python pyserini_indexer.py 
"""

import os.path
import json
import pickle


if __name__ == "__main__":
    

    index_path = f"cache/indexes/pubmed"

    if not os.path.exists(index_path):
        print('Calling pyserini')
        os.system(f'python -m pyserini.index --input data/pubmed/ --collection JsonCollection --generator DefaultLuceneDocumentGenerator --index {index_path} --threads 1 --storePositions --storeDocvectors --storeRaw')
    else:
        print(f"Anserini index already exist in {index_path}. No action performed")