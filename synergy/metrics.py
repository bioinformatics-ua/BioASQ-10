from polus.metrics import IMetric
from collections import defaultdict


def precision(binary_relevance):
    """
    documents : (score, relevance)
    """    
    if len(binary_relevance)==0:
        return 0 # assume 0
    
    return sum(binary_relevance)/len(binary_relevance)


def average_precision(docs, goldstandard, at=10):

    
    binary_relevance = [ 1 if d_id in goldstandard else 0 for d_id in docs[:at] ]
    precision_at = [precision(binary_relevance[:i]) for i in range(1,at+1)]
    
    return sum([a*b for a,b in zip(precision_at, binary_relevance)])/min(len(goldstandard), at)


def recall(docs, goldstandard, at=10):
    
    id_docs=list(map(lambda x: x[1], docs[:at]))
    
    binary_relevance = [1 if d_id in id_docs else 0 for d_id in goldstandard]
    return precision(binary_relevance)


def map_ir(prediction, goldstandard, at=10):
    """
    Args:
      prediction (dict): A dictionary of predictions with the format
        {query_id:[doc_ids]}, the list of doc_ids should already be sorted by
        document relevance score.

      goldstandard (dict): A dictionary of the goldstandard data with the format
        {query_id:[doc_ids]}.
    """

    aps = [average_precision(docs, goldstandard[q_id], at=at) for q_id, docs in prediction.items() ]
        
    return sum(aps)/len(prediction)



class MAP(IMetric):
    
    def __init__(self, goldstandard, at=10, bioasq=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()
        # argument bioasq it is probably deprecated, remove in the future
        assert (bioasq and at==10) or not bioasq
        self.at = at
        self.bioasq = bioasq
        self.goldstandard = goldstandard
    
    def _samples_from_batch(self, samples):
        for q, s, d in zip(samples["query_id"], samples["score"], samples["doc_id"]):
            self.current_ranking_order[q.numpy().decode()].append((s.numpy(), d.numpy().decode()))
    
    def reset(self):
        self.current_ranking_order = defaultdict(list)
    
    def _evaluate(self):
        # this line can be avoided if instead of a defaultdict(list) we use a sorted dict
        # where each insert already places the document into its right place
        prediction = {q_id:[_id for s,_id in sorted(docs, key=lambda x:-x[0])] for q_id, docs in self.current_ranking_order.items()}
        
        # call the map function
        return map_ir(prediction, self.goldstandard, self.at)
    

class Recall(IMetric):
    
    def __init__(self, goldstandard, at=10, bioasq=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()
        assert (bioasq and at==10) or not bioasq
        self.at = at
        self.bioasq = bioasq
        self.goldstandard = goldstandard
    
    def _samples_from_batch(self, samples):
        # This is probably wrong need to check with data format!!
        for q, s, d in zip(samples["query_id"], samples["score"], samples["doc_id"]):
            self.current_ranking_order[q.numpy().decode()].append((s.numpy(), d.numpy().decode()))
    
    def reset(self):
        self.current_ranking_order = defaultdict(list)
    
    def _evaluate(self):
        recalls = [recall(sorted(docs, key=lambda x:-x[0]), self.goldstandard[q_id], at=self.at) for q_id, docs in self.current_ranking_order.items() ]
        
        return sum(recalls)/len(self.current_ranking_order)