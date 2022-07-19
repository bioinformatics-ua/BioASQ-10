"""
Replication test on 19/07/2022 was sucessfull by Tiago Almeida

Command: python process_questions_feedback_data.py
"""


import json
import re
from collections import defaultdict

import argparse

def get_q_ids(x):
    return set(map(lambda x:x["id"], questions[x]))

def convert_dict(x):
    return {y["id"]:y for y in x}

def write_docs_q_rels(bioasq_feedback, output_file_name):
    num_total_docs_rels = 0

    with open(output_file_name, "w") as f:
        
        for data in bioasq_feedback:
            q_id = data["id"]
            
            for docs in data["documents"]:
                doc_rel = 2 if docs["golden"] else 0
                num_total_docs_rels += 1
                doc_id = docs["id"]
                f.write(f"{q_id} None {doc_id} {doc_rel}\n")
                
    print(f"Ouput to file {output_file_name}, total number of documents relevants found {num_total_docs_rels}")
    

def write_queries_in_tsv(queries, output_file_name):
    
    with open(output_file_name, "w") as f:
        for query in queries:
            q_id = query["id"]
            q_text = query["body"]
            f.write(f"{q_id}\t{q_text}\n")
                     

def write_goldstandard(test_set, bioasq_feedback, bioasq_feedback_next_round, output_name):
    
    bioasq_feedback_next_round = convert_dict(bioasq_feedback_next_round)
    bioasq_feedback = convert_dict(bioasq_feedback)
    
    for data in test_set:
        
        if data["id"] in bioasq_feedback_next_round:
            
            previous_round_docs = set(map(lambda x:x["id"], bioasq_feedback[data["id"]]["documents"]))
            
            data["documents"] = [ doc for doc in bioasq_feedback_next_round[data["id"]]["documents"] if doc["id"] not in previous_round_docs]
        else:
            data["documents"] = []

            
    with open(output_name, "w") as f:
        json.dump({"questions":test_set}, f)
    
def build_n_grams(l, ngram=3):
    return {"-".join(l[i:i+ngram]) for i in range(len(l)-ngram+1)}

def approx_finder(text, pattern):
    text_tokens = text.split(" ")
    pattern_tokens = pattern.split(" ")
    
    n_grams_text = build_n_grams(text_tokens, 4) | build_n_grams(text_tokens, 3)
    n_grams_pattern = build_n_grams(pattern_tokens, 4) | build_n_grams(pattern_tokens, 3)
    
    if len(n_grams_pattern)>0:
        text_approx_contains_pattern = 1 - len(n_grams_pattern-n_grams_text)/len(n_grams_pattern)
    else: 
        text_approx_contains_pattern = 0
    if len(n_grams_text)>0:
        pattern_approx_contains_text = 1 - len(n_grams_text-n_grams_pattern)/len(n_grams_text)
    else:
        pattern_approx_contains_text = 0
    
    return max(text_approx_contains_pattern, pattern_approx_contains_text)
    
def load_collection_from_jsonl(jsonl_file):
    collection = None #defaultdict(dict)
    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if "_" in data["id"]:
                if collection is None:
                    collection = defaultdict(dict)
                doc_id, sentence_id = data["id"].split("_")
                collection[doc_id][sentence_id] = data["contents"]
            else:
                if collection is None:
                    collection = {}
                collection[data["id"]] = data["contents"]
                
            
    return collection
    
def get_snippet_ids(collection, snippet, t=0.8):
    """
    Find the correct snippets ids from the documents,
    note that a gold snippet can fit more that one sentence.
    
    collection: dict <doc_id,<sentence_id, text>>
    """
    rel_sentences = list()
    
    doc_id = snippet["document"]
    for sentence_id, sentence in collection[doc_id].items():
        if approx_finder(sentence, snippet["text"]) > t:
            rel_sentences.append(f"{doc_id}_{sentence_id}")
    
    return rel_sentences


def write_snippets_q_rels(bioasq_feedback, collection, output_file_name):
    num_total_snippets_rels = 0
    
    with open(output_file_name, "w") as f:
        
        for data in bioasq_feedback:
            q_id = data["id"]
            
            for snippet in data["snippets"]:
                for doc_col_id in get_snippet_ids(collection, snippet):
                    doc_rel = 2 
                    num_total_snippets_rels += 1
                    f.write(f"{q_id} None {doc_col_id} {doc_rel}\n")
                
    print(f"Ouput to file {output_file_name}, total number of snippets relevants found {num_total_snippets_rels}")



def read_synergy_dataset(question_base_path, feedback_base_path):
    
    questions = {}
    feedback = {}
    cumulative_ids = set()
    cumulative_feedback_ids = set()
    for rnd in [1,2,3,4]:
        with open(f"{question_base_path}{rnd}", "r") as f:
            questions[rnd] = json.load(f)["questions"]
        with open(f"{feedback_base_path}{rnd}", "r") as f:
            feedback[rnd] = json.load(f)["questions"]

            print(f"Questions in round {rnd}: {len(questions[rnd])}", end=" | ")
            rnd_q_ids = set(map(lambda x:x["id"], questions[rnd]))
            print(f"Number of new questions: {len(rnd_q_ids - cumulative_ids)}", end=" | ")
            cumulative_ids |= rnd_q_ids

            feedback_rnd_q_ids = {x["id"] for x in feedback[rnd] if len(x["documents"])>0}
            print(f"Number of questions without feedback data: {len(rnd_q_ids - feedback_rnd_q_ids)}", end=" | ")

            total_docs_relevant = sum(map(lambda x: len(x["documents"]), feedback[rnd]))
            print(f"Total number of feedback documents (0 or 1) in the feedback:", total_docs_relevant)

    return questions, feedback, cumulative_ids, cumulative_feedback_ids

if __name__ == "__main__":
    
    base_path = "data/dataset/BioASQ_Synergy10"
    
    data = read_synergy_dataset(f"{base_path}/BioASQ-taskSynergy_v2022-testset",
                                f"{base_path}/BioASQ-taskSynergy_2022-feedback_round")
    
    questions, feedback, cumulative_ids, cumulative_feedback_ids = data
    
    # preprocessing for round 3 and 4 
    rounds = [3, 4]

    collection_base_path = "cache/clean_collection"

    for rnd in rounds:
        write_docs_q_rels(feedback[rnd], f"{base_path}/BioASQ-taskSynergy_2022_documents_feedback_round{rnd}_trecformat.txt")
        write_queries_in_tsv(questions[rnd], f"{base_path}/BioASQ-taskSynergy_2022_testset{rnd}_trecformat.tsv")
        if rnd<len(rounds):
            write_goldstandard(questions[rnd], feedback[rnd], feedback[rnd+1], f"BioASQ-taskSynergy_2022_goldstandard_round{rnd}.json")
        if rnd>1:
            for f_col in ["merge", "nomerge"]:
                collection = load_collection_from_jsonl(f"{collection_base_path}/rnd{rnd}/{f_col}/collection.jsonl")
                write_snippets_q_rels(feedback[rnd], collection, f"{base_path}/BioASQ-taskSynergy_2022_snippets_feedback_round{rnd}_{f_col}_trecformat.txt")
            