from transformers import AutoTokenizer
from nltk.corpus import sentiwordnet as swn
import nltk
import random



def _jumble(quest, snipp, rand):
    [rand.shuffle(x) for x in snipp]
    final_snipp=list(zip(*snipp))
    return ({x[1]:final_snipp[x[0]] for x in enumerate(quest)})

def _pairing_with_para(quest, paraphrase_q, paraphrase_s, rand):
        pairs=list()

        for q in quest:
            ql = [q['body']]+paraphrase_q[q['id']]

            sl =  list(zip(*([tuple([s["text"] for s in q['snippets']])] + list(zip(*paraphrase_s[q['id']])))))
            sl = [list(x) for x in sl]

            question_sent_matrix = _jumble(ql,sl,rand)
            tmp = 0 if q['exact_answer'] == "no" else 1 
            for i,qs_q in enumerate(question_sent_matrix):
                for qs_s in question_sent_matrix[qs_q]:
                    pairs.append((qs_q, qs_s, tmp, f"{q['id']}|{i}"))

        rand.shuffle(pairs)
        return pairs
        
def _pairing(quest,rand):
    pairs=list()
    for q in quest:
        tmp = 0 if q['exact_answer'] == "no" else 1

        for s in q['snippets']:
            pairs.append((q['body'], s['text'], tmp, q['id']))

    rand.shuffle(pairs)
    return pairs



def question_snippet(entries, seed, paraphrase_q, paraphrase_s, model):
    rand = random.Random(seed)
    def generator():
        if (paraphrase_q is not None) and (paraphrase_s is not None):
            paired_entries = _pairing_with_para(entries, paraphrase_q, paraphrase_s,rand)
        else:
            paired_entries = _pairing(entries,rand)

        tokenizer = AutoTokenizer.from_pretrained(model['transformer_checkpoint'])
        for e in paired_entries:
            tmp_tk=tokenizer.encode_plus((e[0],e[1]),truncation=True,padding='max_length',max_length=128)
            yield{
                'question_id': e[3],
                'input_ids':tmp_tk['input_ids'],
                'token_type_ids':tmp_tk['token_type_ids'],
                'attention_mask':tmp_tk['attention_mask'],
                'label':e[2]
            }
    return generator



# Get positive and negative score to a word using sentwordnet 
def _calculate_word_sentiment(word):
    results = list(swn.senti_synsets(word))
    pos_score_sum=0
    neg_score_sum=0
    
    if len(results) == 0:
        return 0, 0

    for res in results:
        pos_score_sum+=res.pos_score()
        neg_score_sum+=res.neg_score()

    pos = pos_score_sum/len(results)
    neg = neg_score_sum/len(results)
    
    return pos, neg


#Give score between -1 and 1 to the sentence
def _calculate_sent_sentiment(sentence):
    acc_score = 0
    
    for word:=word.strip() in sentence.split(' '):
        pos_score, neg_score = _calculate_word_sentiment(word)
        
        
        #pos and neg are values between 0 and 1, so their subtraction will give a value between -1 and 1
        acc_score = pos_score - neg_score
        
    return acc_score/len(sentence)


def question_snippet_with_sents(entries, model):
    nltk.download('wordnet')
    nltk.download('sentiwordnet')
    rand = random.Random(seed)
    def generator():
        if (paraphrase_q is not None) and (paraphrase_s is not None):
            paired_entries = _pairing_with_para(entries, paraphrase_q, paraphrase_s, rand)
        else:
            paired_entries = _pairing(entries,rand)

        tokenizer = AutoTokenizer.from_pretrained(model['transformer_checkpoint'])
        for e in paired_entries:
            tmp_tk=tokenizer.encode_plus((e[0],e[1]),truncation=True,padding='max_length',max_length=128)
            ss = _calculate_sent_sentiment(e[1])
            yield{
                'question_id': e[3],
                'input_ids':tmp_tk['input_ids'],
                'token_type_ids':tmp_tk['token_type_ids'],
                'attention_mask':tmp_tk['attention_mask'],
                'sent': ss,
                'label':e[2]
            }
    return generator
