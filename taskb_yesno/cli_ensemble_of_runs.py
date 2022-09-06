import os
os.environ["POLUS_JIT"] = "false"
from polus.models import load_model
from polus.data import DataLoader
import argparse
import json
from transformers import AutoTokenizer
import tensorflow as tf

import models as MODELS

def test_data_gen(question_list):
    
    def generator():
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        for q in question_list["questions"]:
            if q["type"] == "yesno":
                for s in q['snippets']:
                    tmp_tk=tokenizer.encode_plus((q["body"],s["text"]),truncation=True,padding='max_length',max_length=128)
                    yield{
                        'question_id': q["id"],
                        'input_ids':tmp_tk['input_ids'],
                        'token_type_ids':tmp_tk['token_type_ids'],
                        'attention_mask':tmp_tk['attention_mask']
                    } 

    return generator

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='YES NO Training script')
    parser.add_argument("--runs", nargs="+", type=str)
    parser.add_argument("--testset_path", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)

    path_ensemble
    args = parser.parse_args()
    
    print(args.runs)
    
    with open(args.testset_path) as f:
        testset = json.load(f)
    
    to_dict = lambda x: {data["id"]:data for data in x}
    
    to_int = lambda x: 1 if x=="yes" else 0
    
    for q_data in testset["questions"]:
        for run_path in args.runs:
            with open(run_path) as f:
                json_run = to_dict(json.load(f)["questions"])
        if q_data["type"] == "yesno":
            
            run_guess = to_int(json_run[q_data["id"]]["exact_answer"])
            
            if "exact_answer" in q_data:
                q_data["exact_answer"].append(run_guess)
            else:
                q_data["exact_answer"] = [run_guess]
                
    # majority voting
    for q_data in testset["questions"]:
        if q_data["type"] == "yesno":
            q_data["exact_answer"] = "yes" if sum(q_data["exact_answer"])>(len(q_data["exact_answer"])/2) else "no"
            q_data["ideal_answer"] = ""
        else:
            q_data["exact_answer"] = ""
            q_data["ideal_answer"] = ""
        
    #print(solved_test)
    
    #path_ensemble = "|".join(map(lambda x: os.path.basename(x), args.runs)) + ".json"
    path_ensemble = args,save_path
    with open(path_ensemble, 'w') as f:
        json.dump(testset, f)
    
