from collections import defaultdict
from utils import split_title_abstract, default_to_regular

def span_interssection(spanA, spanB):
    return spanA[2]==spanB[2] and not(spanA[1] < spanB[0] or spanB[1] < spanA[0])

def get_span(x):
    return (x["start"],x["end"],int(x["section"]=="title"))

def get_similar_span_in_dict(d_keys, span):

    for k in d_keys:
        if span_interssection(k, span):
            return k 
    return None

def rearrange_and_clean_snippets(query_data_list):
    for query_data in query_data_list:
        query_data["snippets"] = rearrange_and_clean_snippets_list(query_data["snippets"])
        
    return query_data_list

def rearrange_and_clean_snippets_list(snippet_list):
    d = defaultdict(dict)
    for snippet in snippet_list:
        span = get_span(snippet)
        key_span = get_similar_span_in_dict(d[snippet["doc_id"]].keys(), span)
        if key_span is not None: # found
            if (key_span[1]-key_span[0])>(span[1]-span[0]):
                # update the dict if the len of span is smaller that len of key_span
                del d[snippet["doc_id"]][key_span]
                d[snippet["doc_id"]][span] = snippet
        else:
            d[snippet["doc_id"]][span] = snippet
    
    # re-convert to dict of list of spans
    out_d = defaultdict(list)
    for k,v in d.items():
        for _,s in v.items():
            #print(s)
            out_d[k].append(s)
        
    return default_to_regular(out_d)

def convert_to_span_section_format(snippet_data):
    return {k:list(map(lambda x:(x["start"],x["end"],int(x["section"]=="title")), v)) for k,v in snippet_data.items()}