import os
os.environ["POLUS_JIT"] = "false"
from polus.models import load_model
from polus.data import DataLoader
import argparse
import json
from transformers import AutoTokenizer
import tensorflow as tf
from functools import reduce

import models as MODELS

from cli_train import make_inference




class Inferencer:
    def __init__(self):
        self.answers = dict()

    def process_batch(self,sample):
        for i in range(len(sample["question_id"])):
            q_id = samples["question_id"][i].numpy().decode()
            if q_id not in self.answers.keys():
                self.answers[q_id] = []
            self.answers[q_id].append(samples["y_pred"][i].numpy())


    def get_results(self):
        y_pred = dict()
        for q_id in self.answers.keys():

            #Majority voting (wrong?)
            #y_pred[q_id] = "yes" if  sum(self.answers[q_id])>=(len(self.answers[q_id])/2) >=0.5 else "no" 
            #maj_wrong = "yes" if  sum(self.answers[q_id])>=(len(self.answers[q_id])/2) >=0.5 else "no" 

            #Majority voting (right?)
            #maj_right = "yes" if reduce(lambda x,y: x+1 if y>=0.5 else x,self.answers[q_id])>=len(self.answers[q_id])/2 else 'no'

            #Highest confidence
            confidences = list(map(lambda x: (x,1.0) if x>=0.5 else (1-x,0.0),self.answers[q_id]))
            highest_confidence = max(confidences,key=lambda x: x[0])
            y_pred[q_id]='yes' if highest_confidence[1]==1.0 else 'no' 

            #print(maj_wrong,maj_right,highest_confidence[1])
          
        return y_pred
    
        
        

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
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--testset_path", default="/tf/volume/BioASQ-10b/yes_no/BioASQ-task10bPhaseB-testset2.json", type=str)
    args = parser.parse_args()
    
    model = load_model(args.model_path, external_module=MODELS)
    
    original_test_json = json.load(open(args.testset_path ,'r'))
    
    
    gen_test = test_data_gen(original_test_json)
    
    
    test_dl = DataLoader(source_generator=gen_test)
    
    test_tfds = val_dl = CachedDataLoader.from_cached_index(os.path.join(PATH_CACHE, "dataloaders", cfg["generators"]["validation"]))
        
    inf = Inferencer()
    for step, samples in enumerate(test_tfds):
        #print(samples)
        s = make_inference(model, samples)
        #print(s)
        inf.process_batch(s)
    
    results = inf.get_results() 
    
    solved_test = original_test_json
    del original_test_json
    
    
    for q in solved_test["questions"]:
        if q["id"] in results:
            q["exact_answer"] = results[q["id"]]
            q["ideal_answer"] = ""
        else:
            q["exact_answer"] = ""
            q["ideal_answer"] = ""
    
    #print(solved_test)
    basename=os.path.basename(args.model_path).split(".")[0]
    with open(f'{basename}.json', 'w') as f:
        json.dump(solved_test, f)
  
