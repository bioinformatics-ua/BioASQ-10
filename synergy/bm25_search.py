import argparse
import os
import pyserini
import json
from metrics import map_ir
from collections import defaultdict
import copy
import pathlib

def run_search(query_file, 
               anserini_index, 
               relevance_feedback_file = None,
               name= "",
               k1=0.9,
               b=0.4,
               fb_terms=100,
               fb_docs=10,
               originalQW=0.5):
    
    if name == "":
        _index_name = pathlib.PurePath(anserini_index).name
    
        print("INDEX_NAME", _index_name)

        output_file = "runs/" + os.path.splitext(os.path.basename(query_file))[0]+"_" + _index_name + ".txt"
    else:
        output_file = "runs/" + name + ".txt"
    
    anserini_jar = f"{os.path.dirname(pyserini.__file__)}/resources/jars/anserini-0.14.0-fatjar.jar"
    cmd_str = f"java -cp {anserini_jar} io.anserini.search.SearchCollection"
    
    if relevance_feedback_file is not None:
        # -rm3.outputQuery add this argument to see the augmented rm3 query
        os.system(f'{cmd_str} -index {anserini_index} ' +
                  f'-topicreader TsvString -topics {query_file} -removedups ' +
                  f'-bm25 -bm25.k1 {k1} -bm25.b {b} ' +
                  f'-rm3 -rm3.fbDocs {fb_docs} -rm3.fbTerms {fb_terms} -rm3.originalQueryWeight {originalQW} ' +
                  f'-hits 300  -rf.qrels {relevance_feedback_file} ' +
                  f'-output {output_file} ')
    else:
        os.system(f'{cmd_str} -index {anserini_index} ' +
                  f'-topicreader TsvString -topics {query_file} -removedups ' +
                  f'-bm25 -bm25.k1 {k1} -bm25.b {b} ' +
                  f'-rm3 -rm3.fbDocs {fb_docs} -rm3.fbTerms {fb_terms} -rm3.originalQueryWeight {originalQW} ' +
                  f'-hits 300 ' +
                  f'-output {output_file}')
        
    print("Results on", output_file)
    return output_file

def convert_trec_to_bioasq(run_trec_file):
    
    bioasq_run = defaultdict(list)
    
    with open(run_trec_file, "r") as f:
        for line in f:
            line = line.split(" ")
            # score in line[4]
            bioasq_run[line[0]].append(line[2])
            
    return bioasq_run

def load_qrels(qrels_file):
    qrels = defaultdict(list)
    with open(qrels_file, "r") as f:
        for line in f:
            line = line.split(" ")
            if int(line[3])>0:
                qrels[line[0]].append(line[2])
                
    return qrels

def remove_docs_from_previous_rounds(bioasq_run, relevance_feedback_file):
    qrels = convert_trec_to_bioasq(relevance_feedback_file)
    
    for k in bioasq_run.keys():
        if k in qrels:
            bioasq_run[k] = [ doc for doc in bioasq_run[k] if doc.split("_")[0] not in qrels[k]]
    
    
    return bioasq_run

def maybe_convert_to_rank_by_doc_id(bioasq_run):
    
    _k = list(bioasq_run.keys())[0]
    
    if "_" in bioasq_run[_k][0]:    
        for q_id in bioasq_run.keys():
            _doc_set = set()
            new_rank = []
            for doc in bioasq_run[q_id]:
                _doc = doc.split("_")[0]
                if _doc not in _doc_set:
                    
                    _doc_set.add(_doc)
                    new_rank.append(_doc)
            
            bioasq_run[q_id] = new_rank
                
    return bioasq_run

def load_trec_run_to_bioasq(run_file, relevance_feedback_file):
    print("LOAD!!!", run_file)
    bioasq_run = convert_trec_to_bioasq(run_file)
    
    if relevance_feedback_file is not None:
        bioasq_run = remove_docs_from_previous_rounds(bioasq_run, relevance_feedback_file)
    
    # filter convert ranked doc_sentence_id to -> doc_id
    # we assume that the score of the top retrieved sentence represents the document
    bioasq_run = maybe_convert_to_rank_by_doc_id(bioasq_run)
    
    return bioasq_run

def evaluate_run(run_file, goldstandard, relevance_feedback_file=None, docs_feedback=None):

    # remove empty GS
    _before = len(goldstandard)
    goldstandard = {k:v for k,v in goldstandard.items() if len(v)>0}
    print(f"A total of {_before-len(goldstandard)} GS questions were removed due to lack of positive feedback")
    
    if docs_feedback is not None:
        bioasq_run = load_trec_run_to_bioasq(run_file, docs_feedback)
    else:
        bioasq_run = load_trec_run_to_bioasq(run_file, relevance_feedback_file)

    # assumption only question with gs will be validated
    _before = len(bioasq_run)
    bioasq_run = {k:v for k,v in bioasq_run.items() if k in goldstandard}
    print(f"A total of {_before-len(bioasq_run)} questions were removed since they do not appear on the goldstandard")
    
    return map_ir(bioasq_run, goldstandard)

def evalute_full_pipeline(query_file, # ../data/BioASQ_Synergy/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset1_trecformat.tsv
                          anserini_index, # indexes/rnd1-cord-collection/
                          goldstandard_file, #../data/BioASQ_Synergy/BioASQ_Synergy10/BioASQ-taskSynergy_2022_goldstandard_round1.json
                          relevance_feedback_file = None, #../data/BioASQ_Synergy/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round1_trecformat.txt
                          docs_feedback = None,
                          name= "",
                          k1=0.9,
                          b=0.4,
                          fb_terms=100,
                          fb_docs=10,
                          originalQW=0.5):
    
    output_file = run_search(query_file,
                             anserini_index,
                             relevance_feedback_file,
                             name=name,
                             k1=k1,
                             b=b,
                             fb_terms=fb_terms,
                             fb_docs=fb_docs,
                             originalQW=originalQW)
    
    with open(goldstandard_file, "r") as f:
        gs = { data["id"]:[ doc["id"] for doc in data["documents"] if doc["golden"]] for data in json.load(f)["questions"]}
            
    return evaluate_run(output_file, gs, relevance_feedback_file, docs_feedback)


def write_as_bioasq(bioasq_queries, bioasq_run, output_file, max_docs=10):
    
    final_run = copy.deepcopy(bioasq_queries)
    
    for query in final_run:
        
        if query["id"] in bioasq_run:
            query["documents"] = bioasq_run[query["id"]][:max_docs]
        else: 
            query["documents"] = []
        
        if "snippets" not in query:
            query["snippets"] = []
        
        if "exact_answer" not in query:
            if query["type"]=="factoid" or query["type"]=="list":
                query["exact_answer"] = []
            elif query["type"]=="yesno":
                query["exact_answer"] = "yes"
                
        if "ideal_answer" not in query:
            query["ideal_answer"] = ""
        
    
    with open(output_file, "w") as f:
        json.dump({"questions":final_run},f)
        
    print("Results on", output_file)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Anserini Relevance Feedback Finetune')
    parser.add_argument("query_file", type=str, help="TREC formated queires file")
    parser.add_argument("index_name", type=str, help="Name of the anserini index to be used")
    parser.add_argument("-q_bio", default=None, type=str, help="Queries in the BioASQ format")
    parser.add_argument("-rf", default=None, type=str, help="relevance feedback file")
    parser.add_argument("-d_rf", default=None, type=str, help="document relevance feedback file")
    parser.add_argument("-gs", default=None, type=str, help="goldstandard file")
    parser.add_argument("-k1", default=0.9, type=float, help="bm25 k1")
    parser.add_argument("-b", default=0.4, type=float, help="bm25 k1")
    parser.add_argument("-fb_terms", default=100, type=int, help="rm3 terms")
    parser.add_argument("-fb_docs", default=10, type=int, help="rm3 docs")
    parser.add_argument("-qw", default=0.5, type=float, help="rm3 original query weight")
    parser.add_argument("-name", default="", type=str, help="name for the run")
    args = parser.parse_args()
    
    ## EXEMPLO:
    ## python cli_bm25.py ../data/BioASQ_Synergy/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset1_trecformat.tsv indexes/rnd1-cord-collection/ -rf ../data/BioASQ_Synergy/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round1_trecformat.txt -gs ../data/BioASQ_Synergy/BioASQ_Synergy10/BioASQ-taskSynergy_2022_goldstandard_round1.json
        
    
    
    if args.gs is not None:
        with open(args.gs, "r") as f:
            gs = { data["id"]:[ doc["id"] for doc in data["documents"] if doc["golden"]] for data in json.load(f)["questions"]}
            
        print("Score", evalute_full_pipeline(args.query_file, 
                                             args.index_name, 
                                             args.gs, 
                                             args.rf,
                                             args.d_rf,
                                             args.name,
                                             k1=args.k1,
                                             b=args.b,
                                             fb_terms=args.fb_terms,
                                             fb_docs=args.fb_docs,
                                             originalQW=args.qw))
    elif args.q_bio is not None:
        output_file = run_search(args.query_file, 
                                 args.index_name, 
                                 args.rf, 
                                 args.name,
                                 k1=args.k1,
                                 b=args.b,
                                 fb_terms=args.fb_terms,
                                 fb_docs=args.fb_docs,
                                 originalQW=args.qw)
        
        with open(args.q_bio, "r") as f:
            bioasq_queries = json.load(f)["questions"]
         
        if args.d_rf is not None:
            bioasq_run = load_trec_run_to_bioasq(output_file, args.d_rf)
        else:
            bioasq_run = load_trec_run_to_bioasq(output_file, args.rf)
        
        output_file_json = output_file.split(".")[0] + ".json"
        
        write_as_bioasq(bioasq_queries, bioasq_run, output_file_json)
        
        
            
        
    else:
        output_file = run_search(args.query_file, 
                                 args.index_name, 
                                 args.rf, 
                                 args.name,
                                 k1=args.k1,
                                 b=args.b,
                                 fb_terms=args.fb_terms,
                                 fb_docs=args.fb_docs,
                                 originalQW=args.qw)
        
    