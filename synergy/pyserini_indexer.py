"""
Replication test on 14/07/2022 was sucessfull by Tiago Almeida

Command: python pyserini_indexer.py 
"""

import os.path
import json
import pickle

def indexer(synergy_dataset, output_path, is_doc_index=True):
    with open(output_path, "w") as f:
        for doc_id, doc in synergy_dataset.items():
            if is_doc_index:
                data = {
                    "id": doc_id,
                    "contents": " ".join(doc)
                }
                f.write(f"{json.dumps(data)}\n")

            else:
                for i,sent in enumerate(doc):
                    data = {
                        "id": '_'.join([doc_id,str(i)]),
                        "contents": sent 
                    }
                    f.write(f"{json.dumps(data)}\n")

if __name__ == "__main__":
    
    """
    Change here to add rnd2 and rnd1 if you want to experiment with rnd1 and 2 of the synergy
    """
    
    rnd=[('rnd3','2022-01-03'), 
         ('rnd4','2022-01-31')]
    
    index=['doc', 'nomerge', 'merge']
    #index=['merge']

    BASE_DIR='cache'

    for rnd_type in rnd:
        for i in index:
            if i == 'doc':     
                filename=os.path.join(BASE_DIR, f'cord19_quest_{rnd_type[1]}_merge.cache')
                is_doc_index=True
            else:
                filename=os.path.join(BASE_DIR, f'cord19_quest_{rnd_type[1]}_{i}.cache')
                is_doc_index=False

            short_name=rnd_type[0]+'-'+i+'-cord-collection'
            output_path='cache/clean_collection/'+rnd_type[0]+'/'+i+'/collection.jsonl'
            dirname_output_path=os.path.dirname(output_path)

            if not os.path.exists(dirname_output_path):
                os.makedirs(dirname_output_path)
            
            print(filename,output_path)
            _,synergy_dataset = pickle.load(open(filename,'rb'))
            indexer(synergy_dataset,output_path,is_doc_index)

            print('Calling pyserini')
            os.system(f'python -m pyserini.index --input {dirname_output_path} --collection JsonCollection --generator DefaultLuceneDocumentGenerator --index cache/indexes/{short_name} --threads 1 --storePositions --storeDocvectors --storeRaw')