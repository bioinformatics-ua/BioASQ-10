from transformers import AutoTokenizer
import tensorflow as tf
import polus as pl 
import random
import json
from spans_utils import convert_to_span_section_format

def aposteriori_filter(question, docs, collection = None):
    # ensure that only documents that belong to the question baseline will be retrieved
    doc_ids = []
    for doc in docs:
        j_doc = json.loads(doc.raw)
                
        if int(question["baseline"]) in j_doc["years"]:
            doc_ids.append(j_doc["id"])
            if collection is not None:
                collection[j_doc["id"]] = j_doc#["contents"]

    return doc_ids

def bm25_negatives_train(dataset, searcher, top_k=500, collection = None, prefix=""):
    
    def generator():
        count = 0
        for q_data in dataset:
            pos_set = set()
            # add pos to the collections
            
            for docid in q_data["documents"]:
                doc = searcher.doc(docid)
                if doc is not None:
                    pos_set.add(docid)
                    if collection is not None:
                        collection[docid] = json.loads(doc.raw())#["contents"]
                else:
                    count+=1
                    #print("docid:",docid, "not found in the index, QUERY:", q_data["phase"], "id", q_data["id"])
            
            docs = searcher.search(q_data["body"], 500)
            docs_ids = aposteriori_filter(q_data, docs, collection)
            # [.docid .score]
            yield {
                "id": q_data["id"],
                "query_text": q_data["body"],
                "doc_pos_id": list(pos_set),
                "doc_neg_id": [docid for docid in docs_ids if docid not in pos_set]
            }
        print("Number of removed positive docs because were not in the index", count)

    generator.__name__ = f"{prefix}k{top_k}_train_gen"
    return generator

def bm25_inference(dataset, searcher, top_k=500, collection=None, prefix=""):
    
    def generator():
        for q_data in dataset:
            
            docs = searcher.search(q_data["body"], top_k)
            docs_ids = aposteriori_filter(q_data, docs, collection)
            
            for doc_id in docs_ids:
                yield {
                    "query_text" : q_data["body"],
                    "query_id" : q_data["id"],
                    "doc_id" : doc_id,
                }

    generator.__name__ = f"{prefix}k{top_k}_validation_gen"
    return generator

def build_tokenizer(checkpoint,
                    max_passages_len = 120,
                    max_passages = 30):

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    def pad_passages(passages):
        if max_passages-len(passages)>0:
            return tf.concat([passages, tf.zeros((max_passages-len(passages), max_passages_len), dtype=tf.int32)], axis=0)
        else:
            return passages[:max_passages]
    
    def encode_pair(query, document):

        inputs=tokenizer.batch_encode_plus(list(zip([query]*len(document), document)),
                                           padding = "max_length",
                                           truncation = True,
                                           max_length = max_passages_len,
                                           return_attention_mask = True,
                                           return_token_type_ids = True,
                                           return_tensors = "tf",
                                          )

        return {k: pad_passages(v) for k,v in inputs.items()}
        

    _short_name = tokenizer.name_or_path.split("/")[-1]
    tokenizer.__name__ = f"{_short_name}_{len(tokenizer)}_len{max_passages_len}"
    encode_pair.hf_tokenizer = tokenizer

    return encode_pair

class INegativeSampler():

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose(self, sample):
        raise NotImplementedError("method choose should be implemented")

    def add_train_tokenizer(self, sample):
        pass


class UniformNegativeSampler(INegativeSampler):

    def choose(self, sample):
        neg_ids = list(map(bytes.decode,
                           sample["doc_neg_id"].numpy().tolist()))
        
        if len(neg_ids)==0:
            return None
        
        if len(neg_ids)>1:
            rnd_neg_index = random.randint(0, len(neg_ids)-1)
        else:
            rnd_neg_index = 0
            
        return neg_ids[rnd_neg_index]
    
    
class WeakNegativeSampler(UniformNegativeSampler):
    
    def __init__(self,
                 collection,
                 steps_total,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.collection_ids = set(collection.keys())
        self.steps_total = steps_total
        self.current_step = 0
        self._num_weak = 0
        self._num_strong = 0
    
    def choose(self, sample):
        """
        This method tries to sample more weak docs in the begining
        of training and then gradually increase the strength of the
        negatives.
        """
        self.current_step += 1
        
        t = self.current_step/self.steps_total
        
        rnd = random.random()
        
        if rnd>t:
            # select from the weak
            #print("Negative sample from weak")
            self._num_weak += 1
            pos_ids = set(map(bytes.decode,
                           sample["doc_pos_id"].numpy().tolist()))
        
            valid_neg_ids = list(self.collection_ids - pos_ids)
            rnd_neg_index = random.randint(0, len(valid_neg_ids)-1)
            return valid_neg_ids[rnd_neg_index]
        else:
            # select from the strong
            #print("Negative sample from strong")
            self._num_strong += 1
            return super().choose(sample)
        
class WeakNegativeSampler_FromCache(UniformNegativeSampler):
    
    def __init__(self,
                 cls_collection,# this var is only temporary, while cls_collection has part of the generator
                 steps_total,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.cls_collection = cls_collection
        self.steps_total = steps_total
        self.current_step = 0
        self._num_weak = 0
        self._num_strong = 0
    
    def choose(self, sample):
        """
        This method tries to sample more weak docs in the begining
        of training and then gradually increase the strength of the
        negatives.
        """
        self.current_step += 1
        #'doc_pos_id', 'doc_neg_id', 'doc_true_neg_id', 'doc_weak_neg_id'
        #neg_ids = self.cls_collection["doc_neg_id"]
        t = self.current_step/self.steps_total
        q_id = sample["query_id"].numpy().decode()
        rnd = random.random()
        
        if rnd>t:
            # select from the weak
            #print("Negative sample from weak")
            self._num_weak += 1

            rnd_neg_index = random.randint(0, len(self.cls_collection[q_id]["doc_weak_neg_id"])-1)
            
            return self.cls_collection[q_id]["doc_weak_neg_id"][rnd_neg_index]
        else:
            # select from the strong
            #print("Negative sample from strong")
            self._num_strong += 1
            return super().choose(sample)



class DynamicNegativeSampler(INegativeSampler):
    """
    Still building this, and not yet tested

    This should do a lot of magic related with negative sampling
    """
    def __init__(self,
                 model,
                 probability_distribution_f,
                 *args,
                 warmup_epochs=10,
                 update_interval=5,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.warmup_epochs = warmup_epochs
        self.uniform_sampler = UniformNegativeSampler()
        self.update_interval = update_interval
        self.probability_distribution_f = probability_distribution_f

        self.current_epoch = 0
        self.current_cached_ids_per_query = {}

    def update_epoch(self):
        self.current_epoch += 1

    def add_train_doc_tokenizer(self, train_doc_tokenizer):
        self.train_doc_tokenizer = train_doc_tokenizer

    def reorganize_neg_ids(self, sample):

        q_id = sample["query_id"].numpy().decode()
        q_text = sample["query_text"].numpy().decode()

        neg_ids = sample["doc_neg_id"].numpy().tolist()

        def _gen():
            for doc_neg_id in neg_ids:
                yield self.train_doc_tokenizer(q_text, doc_neg_id)

        dl = pl.data.DataLoader(_gen).batch(64)

        scores = []
        for s in dl:
            scores.extend(model(**s).numpy().tolist())

        _, new_ids_order = list(zip*(sorted(zip(scores, neg_ids),
                                            key=lambda x: x[0])))

        self.current_cached_ids_per_query[q_id] = new_ids_order

    def choose(self, sample):
        if self.current_epoch < self.warmup_epochs:
            return self.uniform_sampler.choose(sample)
        elif not self.current_epoch % self.update_interval or not self.current_cached_ids_per_query:
            # recompute the order
            self.reorganize_neg_ids(sample)

        q_id = sample["query_id"].numpy().decode()

        neg_probs = self.probability_distribution_f(self.current_cached_ids_per_query[q_id])


        rnd_neg_index = random.choice(self.current_cached_ids_per_query[q_id], p = neg_probs)

        return self.current_cached_ids_per_query[q_id][rnd_neg_index]


def expand_samples(dataloader):
    
    def generator():        
        
        ds = dataloader.shuffle(dataloader.n_samples)
        
        for samples in ds:
            for pos_id in samples["doc_pos_id"].numpy().tolist():
                yield {
                    "query_text" : samples["query_text"],
                    "query_id" : samples["id"],
                    "pos_selected_id" : pos_id,
                    "doc_pos_id": samples["doc_pos_id"].numpy().tolist(),
                    "doc_neg_id" : samples["doc_neg_id"].numpy().tolist(),
                }
            
    return pl.data.DataLoader(generator, magic_k = -1)

def tokenize_validation_set(dataloader, 
                            collection, 
                            tokenizer):
    
    def generator():
        for sample in dataloader:
            q_text = sample["query_text"].numpy().decode()
            doc_id = sample["doc_id"].numpy().decode()
            inputs = tokenizer(q_text, collection[doc_id])
            
            yield {
                "doc_id": doc_id,
                "query_id": sample["query_id"],
                "input_ids": inputs["input_ids"],
                "token_type_ids": inputs["token_type_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        
    return pl.data.DataLoader(generator, magic_k = -1)

def query_doc_pairs(dataloader,
                    collection,
                    tokenizer,
                    negative_sampler: INegativeSampler,
                    shuffle=True):
    """
    This method returns a polus.Dataloader, where each sample has a query a 
    positive doc and a negative docs, additionally is ensured that 
    within a batch each positive doc is only positive to its respective query.
    This makes it possible that for each query all of the other positive docs 
    can be used as negatives examples, speeding up the training.

    """
    
    def get_model_inputs(query_text, doc_id):
        #for sentence in dataloader.get_lookup_data()[doc_id]:
        encode = tokenizer(query_text, collection[doc_id])

        return (encode["input_ids"],
                encode["token_type_ids"],
                encode["attention_mask"])
    
    negative_sampler.add_train_tokenizer(get_model_inputs)
    
    def generator():
        ds = dataloader

        # random sample of the questions
        if shuffle:
            ds = ds.shuffle(dataloader.get_n_samples())

        for sample in ds:

            # random selected query
            q_text = sample["query_text"].numpy().decode()
            # all of the positive docs for the selected query
            # NOTE: if pos_docs_ids are all unique, use list instead of set
            if "pos_selected_id" not in sample:
                pos_ids = list(map(bytes.decode,
                                   sample["doc_pos_id"].numpy().tolist()))

                rnd_pos_index = random.randint(0, len(pos_ids)-1)
                sample["pos_selected_id"] = pos_ids[rnd_pos_index]
            else:
                sample["pos_selected_id"] = sample["pos_selected_id"].numpy().decode()

            sample["pos_selected_input_ids"], sample["pos_selected_type_ids"], sample["pos_selected_attention_mask"] = get_model_inputs(q_text, sample["pos_selected_id"])

            sample["neg_selected_id"] = negative_sampler.choose(sample)
            if sample["neg_selected_id"] is None:
                continue
            
            sample["neg_selected_input_ids"], sample["neg_selected_type_ids"], sample["neg_selected_attention_mask"] = get_model_inputs(q_text, sample["neg_selected_id"])

            # delete all unused references, it also helps with the batching of uneven samples
            del sample["doc_pos_id"]
            del sample["doc_neg_id"]

            yield sample

    return pl.data.DataLoader(generator)#.batch(batch_size, drop_remainder=True)


def pad_passages_cls(passages_cls, max_passages):
    #passages_cls = tf.constant(passages_cls)
    passages_cls = tf.convert_to_tensor(passages_cls)
    emb_size = len(passages_cls[0])
    
    if max_passages-len(passages_cls)>0:
        return tf.concat([passages_cls, tf.zeros((max_passages-len(passages_cls), emb_size), dtype=tf.float32)], axis=0)
    else:
        return passages_cls[:max_passages]

def get_cls_from_cache(dataloader, cls_collection, max_passages):
    
    def generator():
        for sample in dataloader:
            
            q_id = sample["query_id"].numpy().decode()
            doc_id = sample["doc_id"].numpy().decode()
            sample["cls_embedding"] = pad_passages_cls(cls_collection[q_id]["cls_db"][doc_id], max_passages)
            yield sample
        
    
    return pl.data.DataLoader(generator)
    
def query_doc_pairs_w_cached_cls(dataloader,
                                 cls_collection,
                                 negative_sampler: INegativeSampler,
                                 max_passages = 10,
                                 shuffle=True):
    """
    This method returns a polus.Dataloader, where each sample has a query a 
    positive doc and a negative docs, additionally is ensured that 
    within a batch each positive doc is only positive to its respective query.
    This makes it possible that for each query all of the other positive docs 
    can be used as negatives examples, speeding up the training.

    """
    
    def generator():
        ds = dataloader

        # random sample of the questions
        if shuffle:
            ds = ds.shuffle(dataloader.get_n_samples())

        for sample in ds:

            # random selected query
            q_id = sample["query_id"].numpy().decode()
            # all of the positive docs for the selected query
            # NOTE: if pos_docs_ids are all unique, use list instead of set
            if "pos_selected_id" not in sample:
                pos_ids = list(map(bytes.decode,
                                   sample["doc_pos_id"].numpy().tolist()))

                rnd_pos_index = random.randint(0, len(pos_ids)-1)
                sample["pos_selected_id"] = pos_ids[rnd_pos_index]
            else:
                sample["pos_selected_id"] = sample["pos_selected_id"].numpy().decode()
                
            sample["pos_selected_embeddings"] = pad_passages_cls(cls_collection[q_id]["cls_db"][sample["pos_selected_id"]], max_passages)

            sample["neg_selected_id"] = negative_sampler.choose(sample)
            sample["neg_selected_embeddings"] = pad_passages_cls(cls_collection[q_id]["cls_db"][sample["neg_selected_id"]], max_passages)

            # delete all unused references, it also helps with the batching of uneven samples
            del sample["doc_pos_id"]
            del sample["doc_neg_id"]

            yield sample

    return pl.data.DataLoader(generator)#.batch(batch_size, drop_remainder=True)



def query_doc_pairs_for_snippets(dataloader,
                                original_json_dataset,
                                collection, # collection and original_dataset
                                tokenizer,
                                negative_sampler: INegativeSampler,
                                max_doc_passages = 4,
                                shuffle=True):
    """
    This method returns a polus.Dataloader, where each sample has a query a 
    positive doc and a negative docs, additionally is ensured that 
    within a batch each positive doc is only positive to its respective query.
    This makes it possible that for each query all of the other positive docs 
    can be used as negatives examples, speeding up the training.

    """
    
    dict_dataset = {q_data["id"]:q_data for q_data in original_json_dataset}
    
    def generator():
        ds = dataloader

        # random sample of the questions
        if shuffle:
            ds = ds.shuffle(dataloader.get_n_samples())
        
        for sample in ds:

            # random selected query
            q_text = sample["query_text"].numpy().decode()
            q_id = sample["query_id"].numpy().decode()
            # all of the positive docs for the selected query
            # NOTE: if pos_docs_ids are all unique, use list instead of set
            
            snippet_spans_data = convert_to_span_section_format(dict_dataset[q_id]["snippets"])
            
            if "pos_selected_id" not in sample:
                pos_ids = list(map(bytes.decode,
                                   sample["doc_pos_id"].numpy().tolist()))

                rnd_pos_index = random.randint(0, len(pos_ids)-1)
                sample["doc_id"] = pos_ids[rnd_pos_index]
            else:
                sample["doc_id"] = sample["pos_selected_id"].numpy().decode()
                del sample["pos_selected_id"]
            
            if sample["doc_id"] not in snippet_spans_data:
                # skip this positive doc, since we only want to train with pos docs that have snippets
                continue
            
            gs_snippets = snippet_spans_data[sample["doc_id"]]
            neg_doc_id = negative_sampler.choose(sample)
            # choose the negative
            
            del sample["doc_pos_id"]
            del sample["doc_neg_id"]
            
            #yield a positive sample  
            yield {**sample,**tokenizer.tokenize_query_doc(q_text, collection[sample["doc_id"]], gs_snippets=gs_snippets, max_doc_passages=max_doc_passages)}
            
            ## select a negative sample
            #sample["doc_id"] = neg_doc_id
            
            #if sample["doc_id"] is None:
            #    continue

            #yield {**sample,**tokenizer.tokenize_query_doc(q_text, collection[sample["doc_id"]], gs_snippets=[], max_doc_passages=max_doc_passages)}


    return pl.data.DataLoader(generator)

def query_doc_pairs_cached_for_snippets(dataloader,
                                original_json_dataset,
                                collection, # collection and original_dataset
                                tokenizer,
                                model,
                                negative_sampler: INegativeSampler,
                                max_doc_passages = 4,
                                shuffle=True):
    """
    This method returns a polus.Dataloader, where each sample has a query a 
    positive doc and a negative docs, additionally is ensured that 
    within a batch each positive doc is only positive to its respective query.
    This makes it possible that for each query all of the other positive docs 
    can be used as negatives examples, speeding up the training.

    """
    
    dict_dataset = {q_data["id"]:q_data for q_data in original_json_dataset}
    
    def generator():
        ds = dataloader

        # random sample of the questions
        #if shuffle:
        #    ds = ds.shuffle(dataloader.get_n_samples())
        
        for sample in ds:

            # random selected query
            q_text = sample["query_text"].numpy().decode()
            q_id = sample["query_id"].numpy().decode()
            # all of the positive docs for the selected query
            # NOTE: if pos_docs_ids are all unique, use list instead of set
            
            snippet_spans_data = convert_to_span_section_format(dict_dataset[q_id]["snippets"])
            
            if "pos_selected_id" not in sample:
                pos_ids = list(map(bytes.decode,
                                   sample["doc_pos_id"].numpy().tolist()))

                rnd_pos_index = random.randint(0, len(pos_ids)-1)
                sample["doc_id"] = pos_ids[rnd_pos_index]
            else:
                sample["doc_id"] = sample["pos_selected_id"].numpy().decode()
                del sample["pos_selected_id"]
            
            if sample["doc_id"] not in snippet_spans_data:
                # skip this positive doc, since we only want to train with pos docs that have snippets
                continue
            
            gs_snippets = snippet_spans_data[sample["doc_id"]]
            neg_doc_id = negative_sampler.choose(sample)
            # choose the negative
            
            del sample["doc_pos_id"]
            del sample["doc_neg_id"]
            
            #yield a positive sample  
            yield {**sample,**tokenizer.tokenize_query_doc(q_text, collection[sample["doc_id"]], gs_snippets=gs_snippets, max_doc_passages=max_doc_passages)}
            
            ## select a negative sample
            #sample["doc_id"] = neg_doc_id
            
            #if sample["doc_id"] is None:
            #    continue

            #yield {**sample,**tokenizer.tokenize_query_doc(q_text, collection[sample["doc_id"]], gs_snippets=[], max_doc_passages=max_doc_passages)}


    return generator


def query_doc_pairs_joint(dataloader,
                                original_json_dataset,
                                collection, # collection and original_dataset
                                tokenizer,
                                negative_sampler: INegativeSampler,
                                max_doc_passages = 4,
                                shuffle=True):
    """


    """
    
    dict_dataset = {q_data["id"]:q_data for q_data in original_json_dataset}
    
    def generator():
        ds = dataloader

        # random sample of the questions
        if shuffle:
            ds = ds.shuffle(dataloader.get_n_samples())
        
        for sample in ds:

            # random selected query
            q_text = sample["query_text"].numpy().decode()
            q_id = sample["query_id"].numpy().decode()
            # all of the positive docs for the selected query
            # NOTE: if pos_docs_ids are all unique, use list instead of set
            
            snippet_spans_data = convert_to_span_section_format(dict_dataset[q_id]["snippets"])
            
            if "pos_selected_id" not in sample:
                pos_ids = list(map(bytes.decode,
                                   sample["doc_pos_id"].numpy().tolist()))

                rnd_pos_index = random.randint(0, len(pos_ids)-1)
                sample["pos_selected_id"] = pos_ids[rnd_pos_index]
            else:
                sample["pos_selected_id"] = sample["pos_selected_id"].numpy().decode()
            
            # select neg_doc_id
            sample["neg_selected_id"] = negative_sampler.choose(sample)
            
            if sample["neg_selected_id"] is None:
                continue
            
            if sample["pos_selected_id"] in snippet_spans_data:
                sample["has_snippet"] = 1
                gs_snippets = snippet_spans_data[sample["pos_selected_id"]]
            else:
                sample["has_snippet"] = 0
                gs_snippets = []
            
            del sample["doc_pos_id"]
            del sample["doc_neg_id"]
            
            ## concat dict
            sample = {**sample, 
                      **tokenizer.tokenize_document(collection[sample["pos_selected_id"]], 
                                                    gs_snippets=gs_snippets, 
                                                    max_passages=max_doc_passages,
                                                    prefix="pos_"),
                      **tokenizer.tokenize_document(collection[sample["neg_selected_id"]], 
                                                    gs_snippets=[], 
                                                    max_passages=max_doc_passages,
                                                    prefix="neg_"),
                      **tokenizer.tokenize_question(q_text,
                                                    prefix="query_")}
                
            yield sample

    return pl.data.DataLoader(generator)