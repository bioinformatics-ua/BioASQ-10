from transformers import BertTokenizer, TFBertModel

from mmnrm.utils import set_random_seed, load_neural_model, load_model, flat_list
from mmnrm.dataset import TestCollectionV2, sentence_splitter_builderV2
from mmnrm.evaluation import BioASQ_Evaluator

from collections import defaultdict
import os
import pickle
import numpy as np
import sys
import math
import time
import tensorflow as tf

import argparse

from utils import *

import nltk
nltk.download('punkt')

def train_test_generator_for_model(model):
    
    if "model" in model.savable_config:
        cfg = model.savable_config["model"]
    
    max_passages = cfg["max_passages"]
    max_input_size = cfg["max_input_size"]
    tokenizer = model.tokenizer
    
    def maybe_tokenize_pad(query,document):
        if "tokens" not in document:
            input_sentences = []
            sentences =  nltk.sent_tokenize(document["text"])[:max_passages]
            
            for sentence in sentences:
                input_sentences.append([query, sentence])
                
            document["sentences_mask"] = [True]*len(sentences)+[False]*(max_passages-len(sentences))
            
            #pad
            input_sentences.extend([""]*(max_passages-len(sentences)))

            encoded_sentences = tokenizer.batch_encode_plus(
                      input_sentences,
                      max_length=max_input_size,
                      truncation=True,
                      add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                      return_token_type_ids=True,
                      padding="max_length",
                      return_attention_mask=True,
                      return_tensors='np',  # Return tf tensors
                )
            document["tokens"] = encoded_sentences
    
    def train_generator(data_generator):
        
        for query, pos_docs, neg_docs in data_generator:
            
            pos_input_ids = []
            pos_input_masks = []
            pos_input_segments = []
            pos_input_mask_sentences = []
                                                                        
            neg_input_ids = []
            neg_input_masks = []
            neg_input_segments = []
            neg_input_mask_sentences = []                                                            
                                                                    
            for i in range(len(query)):
                pos_doc = pos_docs[i]
                neg_doc = neg_docs[i]
                maybe_tokenize_pad(query[i], pos_doc)
                maybe_tokenize_pad(query[i], neg_doc)
                
                pos_input_ids.append(pos_doc["tokens"]["input_ids"])
                pos_input_masks.append(pos_doc["tokens"]["attention_mask"])
                pos_input_segments.append(pos_doc["tokens"]["token_type_ids"])
                pos_input_mask_sentences.append(pos_doc["sentences_mask"]) 
                                                                    
                neg_input_ids.append(neg_doc["tokens"]["input_ids"])
                neg_input_masks.append(neg_doc["tokens"]["attention_mask"])
                neg_input_segments.append(neg_doc["tokens"]["token_type_ids"])
                neg_input_mask_sentences.append(neg_doc["sentences_mask"]) 
                                                                        
            yield  [tf.convert_to_tensor(np.array(pos_input_ids, dtype="int32"), dtype=tf.int32), 
                    tf.convert_to_tensor(np.array(pos_input_masks, dtype="int32"), dtype=tf.int32),
                    tf.convert_to_tensor(np.array(pos_input_segments, dtype="int32"), dtype=tf.int32),
                    tf.convert_to_tensor(np.array(pos_input_mask_sentences, dtype="bool"), dtype=tf.bool)],\
                   [tf.convert_to_tensor(np.array(neg_input_ids, dtype="int32"), dtype=tf.int32), 
                    tf.convert_to_tensor(np.array(neg_input_masks, dtype="int32"), dtype=tf.int32),
                    tf.convert_to_tensor(np.array(neg_input_segments, dtype="int32"), dtype=tf.int32),
                    tf.convert_to_tensor(np.array(neg_input_mask_sentences, dtype="bool"), dtype=tf.bool)]
    
    def test_generator(data_generator):
        
        for ids, queries, docs in data_generator:
        
            input_query_ids = []

            input_ids = []
            input_masks = []
            input_segments = []

            input_mask_sentences = []
            docs_ids = []

            for i in range(len(ids)):
                for doc in docs[i]:
                    maybe_tokenize_pad(queries[i], doc)
                    input_mask_sentences.append(doc["sentences_mask"])
                    input_ids.append(doc["tokens"]["input_ids"])
                    input_masks.append(doc["tokens"]["attention_mask"])
                    input_segments.append(doc["tokens"]["token_type_ids"])
                    docs_ids.append(doc["id"])
                    input_query_ids.append(ids[i])

            yield input_query_ids, [tf.convert_to_tensor(np.array(input_ids, dtype="int32"), dtype=tf.int32), 
                                    tf.convert_to_tensor(np.array(input_masks, dtype="int32"), dtype=tf.int32),
                                    tf.convert_to_tensor(np.array(input_segments, dtype="int32"), dtype=tf.int32),
                                    tf.convert_to_tensor(np.array(input_mask_sentences, dtype="bool"), dtype=tf.bool)], docs_ids, None
    
    return train_generator, test_generator




def rank(model, t_collection):

    generator_Y = t_collection.generator()
                
    q_scores = defaultdict(list)

    for query_id, Y, docs_ids, offsets_docs in generator_Y:
        s_time = time.time()
        
        scores = model.predict(Y)
        scores = scores[:,0].tolist()
        
        for i in range(len(scores)):
            
            #q_scores[query_id].extend(list(zip(docs_ids,scores)))
            q_scores[query_id[i]].append({"id":docs_ids[i],
                                          "score":scores[i]})
        
        print("\rEvaluation {} | time {}".format(len(q_scores), time.time()-s_time), end="\r")

    # sort the rankings
    for query_id in q_scores.keys():
        q_scores[query_id].sort(key=lambda x:-x["score"])
        q_scores[query_id] = q_scores[query_id]
    
    return q_scores

def rerank_run(baseline_file, top_k):
    run = load_document_run(baseline_file, dict_format=True)

    tCollection = TestCollectionV2(queries, run)\
                      .batch_size(top_k)\
                      .set_transform_inputs_fn(test_input_generator)
    
    results = rank(ranking_model, tCollection)
    
    return create_document_run(queries, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is program will perform pairwise training')
    parser.add_argument('testsets', type=str, help="bioasq round")
    parser.add_argument('baseline_run', type=str, help="path to the baseline run")
    parser.add_argument('model_path', type=str, help="model path")
    parser.add_argument('-o', default="../runs/run",type=str, help="output path")
    
    args = parser.parse_args()
    
    print("load queries")
    
    queries = load_queries(args.testsets, maps=[("body","query")])
    
    print("load model")
    ranking_model = load_model(args.model_path)
    
    print("build generator")
    _, test_input_generator = train_test_generator_for_model(ranking_model)

    print("Start rerank")

    rerank = rerank_run(args.baseline_run, 100)
    print([len(q["documents"]) for q in rerank], min([len(q["documents"]) for q in rerank]), len(rerank))
    
    
    print("write the resutls")
    
    write_as_bioasq(rerank, f"{args.o}.json")
    write_as_trec(rerank, f"{args.o}.trec")
    print("write done", f"{args.o}.trec")