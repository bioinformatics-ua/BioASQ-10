from collections import defaultdict

def default_to_regular(d):
    """
    from https://stackoverflow.com/questions/26496831/how-to-convert-defaultdict-of-defaultdicts-of-defaultdicts-to-dict-of-dicts-o
    """
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

def spans_to_range_dict(span_list):
    d = {}
    for span in span_list:
        for i in range(span[0], span[1]):
            d[i] = span
    return d

def split_title_abstract(article):

    return {"title": article["title"].replace("\n"," "), 
            "abstract": article["contents"][len(article["title"])+2:].replace("\n"," ")}

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)

def scoring_min_diff_between_tokens(sequence_of_words, original_span, valid_tokens):
    # min diff between tokens
    
    #return sum([(sequence_of_words[i][0] - sequence_of_words[i-1][0])**2 for i in range(1,len(sequence_of_words))]) \
            #+ abs(sequence_of_words[0][0]-original_span[0]) + abs(sequence_of_words[-1][0]-original_span[1]) + abs(len(valid_tokens)-len(sequence_of_words))*1000
    return abs(sequence_of_words[0][0]-original_span[0]) + abs(sequence_of_words[-1][0]-original_span[1]) + abs(len(valid_tokens)-len(sequence_of_words))*100000
    

def compute_path_error_spans(result, orignal_span, valid_tokens, error_function=scoring_min_diff_between_tokens):
    #print(result)
    error_spans = [(i,(x[0][0], x[-1][0]+len(x[-1][1])), error_function(x, orignal_span, valid_tokens)) for i, x in enumerate(result)]
    #error = [(i, abs(s-orignal_span[0]) + abs(len(valid_tokens)-l)*1000) for i, (s, e, l) in enumerate(spans)]
    
    return error_spans

def get_min_error_path(result, orignal_span, valid_tokens):
    result = list(filter(lambda x: len(x)>0, result))
    error_spans = compute_path_error_spans(result, orignal_span, valid_tokens)
    
    min_error_span = min(error_spans, key=lambda x:x[2])
    return min_error_span[1], min_error_span[2]
    #return spans[argmin]

        
def correct_snippet_spans_v2(original_abstract, original_s_text, orignal_span, doc_id, threshold = 0.75, topk=2500):
        
    snippet_lower = original_s_text.lower().replace(":"," ").replace("<"," ").replace(">"," ")#.replace("."," ").replace("?"," ").replace("!"," ")
    abstract_lower = original_abstract.lower().replace(":"," ").replace("<"," ").replace(">"," ")#.replace("."," ").replace("?"," ").replace("!"," ")
    
    valid_tokens = list(filter(lambda x: len(x)>0, snippet_lower.split(" ")))
    
    # find all of the occurences of the valid tokens on the abstract
    matches = list()

    for token in valid_tokens:
        match_list = [(m_i, token) for m_i in find_all(abstract_lower, token)]
        if len(match_list)>0:
            matches.append(match_list)
    
    if topk>5000:
        # reduce the keyword search space for large topk
        matches = matches[:30] + matches[-30:]
    
    # cartesian product with constrains
    result = [[]]
    for pool in matches:
        _temp = []
        for x in result:
            for y in pool:
                if len(x)==0 or (y[0]>x[-1][0]):
                    _temp.append(x+[y])
        result.extend(_temp)
        
        # only keep the top-100 most promissing paths
        # ignore the first, bc it is empty
        _result = result[1:]

        errors_spans = compute_path_error_spans(_result, orignal_span, matches)
        errors_spans = sorted(errors_spans, key=lambda x:x[2])
        errors_spans = errors_spans[:topk]

        result = [[]] + [_result[i] for i,s,e in errors_spans]
    
    # remove combinations of tokens that are too short
    result = list(filter(lambda x: len(x)>0 and len(x)/len(matches)>threshold, result))
    
    if len(result) == 0:
        if topk<20000:
            return correct_snippet_spans_v2(original_abstract, original_s_text, orignal_span, doc_id, threshold = 0.75, topk=topk*2)
        #print(f"correct_snippet_spans_v2 didn't find any valid snippet with threshold of {threshold}")
        
        #print(f"{original_abstract=}")
        #print(f"{original_s_text=}")
        #print(f"{orignal_span=}")
        #print(f"{doc_id=}")
        
        return None, None, None
    
    # convert to spans
    min_error_span, score = get_min_error_path(result, orignal_span, valid_tokens)
    final_span = (min_error_span[0], min_error_span[1])
    answer = original_abstract[final_span[0]:final_span[1]]
    
    # correct by ponctuation
    period_index = answer.find(". ")
    if period_index>0 and period_index/len(answer)<0.1:
        # remove the text that is before the ponctuation
        period_index += 2
        final_span = (final_span[0]+period_index, final_span[1])
        answer = original_abstract[final_span[0]:final_span[1]]
        
    return answer, final_span, score
        
def correct_snippet_spans(original_abstract, original_s_text, orignal_span, doc_id, min_len = 1):
    
    if min_len>10:
        print("min word length is to large, so we did not manage to find any match")
        return None, None
    
    snippet_lower = original_s_text.lower().replace("."," ").replace("?"," ").replace("!"," ")
    abstract_lower = original_abstract.lower().replace("."," ").replace("?"," ").replace("!"," ")
    
    tokens = list(filter(lambda x: len(x)>0, snippet_lower.split(" ")))

    valid_tokens = [token for i,token in enumerate(tokens) if len(token)>min_len or i==0 or i==(len(tokens)-1)]
    
    matches = list()
    
    for token in valid_tokens:
        match_list = [m_i for m_i in find_all(abstract_lower, token)]
        if len(match_list)>0:
            matches.append(match_list)
            last_token = token
    pools = [tuple(pool) for pool in matches] 

    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool if len(x)==0 or y>x[-1]]
    
    result = list(filter(lambda x: len(x)>0, result))
    
    if len(result)==0:
        print(f"{original_abstract=}")
        print(f"{original_s_text=}")
        print(f"{abstract_lower=}")
        print(f"{snippet_lower=}")
        print(f"{doc_id=}")
        
        
        print(f"algorithm failed with {min_len}, trying with {min_len+2}")
        raise ValueError()
        return correct_snippet_spans(original_abstract, original_s_text, orignal_span, min_len = min_len+2)
    
    spans = [(x[0], x[-1]+len(last_token)) for x in result]
    
    # find the closest span to the original span
    
    error = [(i, abs(s-orignal_span[0]) + abs(e-orignal_span[1])) for i, (s, e) in enumerate(spans)]
    
    if len(error)==0:
        
        print(f"{original_abstract=}")
        print(f"{original_s_text=}")
        print(f"{abstract_lower=}")
        print(f"{snippet_lower=}")
        print(f"{doc_id=}")
        
        print(f"algorithm failed with {min_len}, trying with {min_len+2}")
        ## debug
        raise ValueError()
        return correct_snippet_spans(original_abstract, original_s_text, orignal_span, min_len = min_len+2)
    
    argmin = min(error, key=lambda x:x[1])[0]
    
    
    final_span = spans[argmin]
    answer = original_abstract[final_span[0]:final_span[1]]
    
    # correct by ponctuation
    period_index = answer.find(". ")
    if period_index>0 and period_index/len(answer)<0.1:
        # remove the text that is before the ponctuation
        period_index += 2
        final_span = (final_span[0]+period_index, final_span[1])
        answer = original_abstract[final_span[0]:final_span[1]]
        
    return answer, final_span