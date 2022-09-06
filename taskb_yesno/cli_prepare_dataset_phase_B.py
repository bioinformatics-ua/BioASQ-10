from data import question_snippet, question_snippet_with_sents
import json
import os
import random
import argparse
from polus.data import CachedDataLoaderwLookup, CachedDataLoader


def metrics_aa(A,B):
    boolA,boolB=[0,0],[0,0]

    for i in A:
        tf=1 if i['exact_answer']=='yes' else 0
        boolA[tf]+=len(i['snippets'])

    for i in B:
        tf=1 if i['exact_answer']=='yes' else 0
        boolB[tf]+=len(i['snippets'])

    totalA=sum(boolA)
    totalB=sum(boolB)
    totals=totalA+totalB
    print('Counts',boolA,boolB)
    print(f'Total Balance: A: {totalA/totals:.2f} B: {totalB/totals:.2f}')
    print(f'False Balance: A: {boolA[0]/totalA:.2f} B: {boolB[0]/totalB:.2f}')

def fair_load_and_split(baselines, split=0.8, downsampling = 1):
    rand=random.Random(42)
    
    filtered_baseline = list(filter(lambda x: x['type']=="yesno" , baselines['questions']))
    
    yes_questions =  list(filter(lambda x: x['exact_answer']=="yes" , filtered_baseline))
    no_questions =  list(filter(lambda x: x['exact_answer']=="no" , filtered_baseline))
    
    del filtered_baseline
    
    
    if len(yes_questions) > len(no_questions):
        yes_questions=yes_questions[:int(len(no_questions) * downsampling) ] 
    elif len(yes_questions) < len(no_questions):
        no_questions = no_questions[:int(len(yes_questions) * downsampling)]
    else:
        ...
    
    all_questions=yes_questions+no_questions
    
    del yes_questions
    del no_questions
    
    rand.shuffle(all_questions)
    
    train_q = all_questions[:int(split*len(all_questions))]
    val_q = all_questions[int(split*len(all_questions)):]

    
    metrics_aa(train_q,val_q)

    return train_q, val_q

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Yes/No data prep script')
    parser.add_argument("--split_perc", type=float, default=0.8, help="The split percentage for training")
    parser.add_argument("--dl_suffix", type=str, default="")
    parser.add_argument("--ds", type=float, default = 1, help="Extra Down sampling")
    args = parser.parse_args()
    
    print(args)
    
    TRAIN_10b_PATH = '../taskb/data/pubmed/training10b.json'
    BERT_CHECKPOINT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

    cfg = {
        "model":{
            "transformer_checkpoint": BERT_CHECKPOINT
        },  
    }
    
    
    
    l_json = json.load(open(TRAIN_10b_PATH ,'r'))
    '''
    yes, no =load_from_json(l_json
    train, val = split(yes, no, args.split_perc)
    '''
    
    #Load paraphrased questions
    paraphrased_quest={}
    with open('paraphrasing/paraphrased_questions.jsonl', 'r') as pj:
        for line in pj:

            j=json.loads(line)
            assert j["paraphrase"]["code"]=="PARAPHRASER_SUCCESS"
            assert j["paraphrase"]["status"]==200
            paraphrased_quest[j["id"]] = [x["alt"] for x in j["paraphrase"]["data"][0]["paras_3"]]    
   
    #Load paraphrased snippets
    paraphrased_snippets={}
    with open('paraphrasing/paraphrased_snippets.jsonl', 'r') as f:
        for l in f:
            j=json.loads(l)
            snippet_list=[]
            for s in j['snippets']: 
                jl=json.loads(s)
                assert jl["code"]=="PARAPHRASER_SUCCESS"
                assert jl["status"]==200
                snippet_list.append([x["alt"] for x in jl["data"][0]["paras_3"]])
            paraphrased_snippets[j['id']]=snippet_list
     
    train, val = fair_load_and_split(l_json,split=args.split_perc, downsampling =args.ds)
    
    print(len(train))
    print(len(val))
    
    
    gen_train = question_snippet(train,42,paraphrased_quest,paraphrased_snippets,**cfg)
    gen_val = question_snippet(val,42,None,None,**cfg)
    
    training_dl = CachedDataLoader(source_generator=gen_train,
                               cache_additional_identifier=f"yesno_split{args.split_perc}_ds{args.ds}_{args.dl_suffix}_training",
                               cache_folder='../taskb/cache/indexes')

    validation_dl =  CachedDataLoader(source_generator=gen_val,
                               cache_additional_identifier=f"yesno_split{args.split_perc}_ds{args.ds}_{args.dl_suffix}_validation",
                               cache_folder='../taskb/cache/indexes')


