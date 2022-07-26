import os

import argparse
import json
from pyserini.search import SimpleSearcher

def replace_keys(data):
    return {
        "id": data["id"],
        "text": data["contents"]
    }

# 'b': 0.6000000000000001, 'fb_docs': 2, 'fb_terms': 16, 'k1': 0.6000000000000001, 'original_query_weight': 0.9

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Build a bm25_baseline.run for the last year model to run')
    # args
    parser.add_argument("test_set_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("-k1", type=float, default=0.6)
    parser.add_argument("-b", type=float, default=0.6)
    parser.add_argument("-fb_terms", type=float, default=16)
    parser.add_argument("-fb_docs", type=float, default=2)
    parser.add_argument("-o_qw", type=float, default=0.9)
    args = parser.parse_args()
    
    with open(args.test_set_path) as f:
        questions = json.load(f)["questions"]
    
    print("init searcher")
    searcher = SimpleSearcher('cache/indexes/pubmed') #pyjni
    print("done")
    #set hyperparams
    searcher.set_bm25(args.k1, args.b)
    searcher.set_rm3(fb_docs=args.fb_docs, fb_terms=args.fb_terms, original_query_weight=args.o_qw)
    
    for q_data in questions:

        hits = searcher.search(q_data["body"], 150)
        
        #for doc in hits:
        #    doc = json.loads(doc.raw)
        #    print(doc["years"])
        #    break
        
        q_data["documents"] = list(map(replace_keys, filter(lambda y: 2022 in y["years"], map(lambda z: json.loads(z.raw), hits))))[:100]
        if len(q_data["documents"])<100:
            print("query",q_data["id"],"has less than 100 docs", len(q_data["documents"]))
            #print(q_data["body"])
        
        
    with open(args.output_path,"w") as f:
        json.dump(questions, f)

        
    print("total num of queires", len(questions))