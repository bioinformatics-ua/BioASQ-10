import argparse
import json
import modelsv2
from data import build_tokenizer
import os
from collections import defaultdict
import spacy

from polus.models import load_model
from polus.data import DataLoader

import copy

def concat_sentences(list_of_sentences):
    
    MAX_FULL_TOKEN_SEQ = 80
    
    new_sents=list()
    
    sent_buf=str()
    sent_buf_size=0
    for sent in list_of_sentences:
        sent_len=len(sent.split(' '))
        if sent_len>MAX_FULL_TOKEN_SEQ:
            #If sentence too large, flush buffer and flush sentence
            new_sents.append(sent_buf)
            sent_buf=str()
            sent_buf_size=0
            new_sents.append(sent)
        elif sent_buf_size+sent_len>MAX_FULL_TOKEN_SEQ:
            #If sentence would overflow buffer, flush buffer add sentence to buffer
            new_sents.append(sent_buf)
            sent_buf=sent
            sent_buf_size=sent_len
        else:
            #Simply add sentence to buffer
            if len(sent_buf)==0:
                sent_buf+=sent
            else:
                sent_buf+=' '+sent
            sent_buf_size+=sent_len
    if sent_buf_size>0: #Flush the buffer one last time
        new_sents.append(sent_buf)
        
    return new_sents

def init_sentence_split():
    nlp = spacy.load('en_core_web_lg')
    def sentence_split(doc_text):
        
        return concat_sentences(map(lambda x:str(x), nlp(doc_text).sents))
    return sentence_split

def get_validation_dataloader(queries, baseline, tokenizer):
    
    sent_split = init_sentence_split()
    
    def generator():
        
        for query in queries:
            for doc in baseline[query["id"]]["documents"]:
                
                # do doc sent split
                inputs = tokenizer(query["body"], sent_split(doc["text"]))
                
                yield {
                    "query_id": query["id"],
                    "doc_id": doc["id"],
                    "input_ids" : inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "token_type_ids": inputs["token_type_ids"]
                }
        
    return DataLoader(generator)
    
def write_as_trec(run, file):
    
    with open(file, "w") as f:
        for q_id, docs in run.items():
            for rank, doc in enumerate(docs):
            #for rank, doc in enumerate(query["documents"]):
                f.write("{} Q0 {} {} {} {}\n".format(q_id,
                                                     doc["doc_id"],
                                                     rank,
                                                     doc["score"],
                                                     "bioasq_as_trec"))
                
def write_as_bioasq(testset, run, file, max_docs=10, max_snippets=10):
    
    final_run = copy.deepcopy(testset)
    
    for query in final_run:
        
        if "query" in query:
            query["body"] = query.pop("query")
        
        query["documents"] = list(map(lambda x: "http://www.ncbi.nlm.nih.gov/pubmed/"+x["doc_id"], run[query["id"]]))[:max_docs]
        
        if "snippets" not in query:
            query["snippets"] = []
        else:
            snippets = []
            for snippet in query["snippets"][:max_snippets]:
                section = "title" if snippet["is_title"] else "abstract"
                snippets.append({
                    "beginSection": section,
                    "endSection": section,
                    "text": snippet["text"],
                    "document": "http://www.ncbi.nlm.nih.gov/pubmed/"+snippet["doc_id"],
                    "offsetInBeginSection": snippet["start"],
                    "offsetInEndSection": snippet["end"]
                })
            query["snippets"] = snippets
            
    with open(file, "w", encoding="utf-8") as f:
        json.dump({"questions":final_run}, f)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument("model_path", type=str, default=str)
    parser.add_argument("test_set", type=str, default=str)
    parser.add_argument("baseline_run", type=str, default=str)
    parser.add_argument("-o", type=str, default="runs")

    args = parser.parse_args()
    
    print("load queries")
    with open(args.test_set) as f:
        queries = json.load(f)["questions"]
        
    print("load baseline run")
    with open(args.baseline_run) as fRun:
        baseline = {q_data["id"]:q_data for q_data in json.load(fRun)}
        
    print("load model")
    model = load_model(args.model_path, external_module=modelsv2)
    cfg = model.savable_config
    
    tokenizer = build_tokenizer(cfg["model"]["checkpoint"], **cfg["tokenizer"])
    
    dataloader = get_validation_dataloader(queries, baseline, tokenizer)
    dataloader = dataloader.batch(64)
    
    ranking_order = defaultdict(list)
    
    print("Make predictions")
    for i, data in enumerate(dataloader):
        print(i, end="\r")
        scores = model(input_ids=data["input_ids"], 
                       attention_mask=data["attention_mask"], 
                       token_type_ids=data["token_type_ids"])
        
        for i in range(scores.shape[0]):
            doc_i = {"score": scores[i].numpy(), "doc_id": data["doc_id"][i].numpy().decode()}
            query_id = data["query_id"][i].numpy().decode()
            
            ranking_order[query_id].append(doc_i)
    
    print("sort rankings")
    # sort
    for q_id in ranking_order.keys():
        ranking_order[q_id] = sorted(ranking_order[q_id], key=lambda x: -x["score"])
        assert ranking_order[q_id][0]["score"] >= ranking_order[q_id][1]["score"]
        
    print("Save as bioasq and trec")
    _name = os.path.splitext(os.path.basename(args.model_path))[0]
    write_as_bioasq(queries, ranking_order, f"{args.o}/{_name}.json")
    write_as_trec(ranking_order, f"{args.o}/{_name}.trec")