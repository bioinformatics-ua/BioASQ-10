from transformers import AutoTokenizer
import spacy
import glob
import sys
import json
import csv
import math
from functools import reduce
import pickle
from os.path import exists,basename,dirname
from multiprocessing import Queue,Process
import fasttext
import re
import click


class SynergyDataSterilizer:
    def __init__(self,
                 synergy_data_path,# = '/tf/volume/data/BioASQ_Synergy/',
                 cord_data_path,# = '/tf/volume/data/Cord-19/2021-12-20/metadata.csv',
                 question_uppercase_first_letter,# = True,
                 inconsistent_rule,# = 'to_false', #'to_true' 'to_false' 'to_unk' 'ignore'
                 max_full_token_seq,# = 80,
                 pool_thread_count,# = 25
                 sentence_merging
                ):
        print(cord_data_path)
        print(question_uppercase_first_letter)
        
        self.SYNERGY_DATA_PATH=synergy_data_path
        self.CORD_DATA_PATH=cord_data_path
        self.QUESTION_UPPERCASE_FIRST_LETTER=question_uppercase_first_letter
        self.INCONSISTENT_RULE=inconsistent_rule
        self.POOL_THREAD_COUNT=pool_thread_count
        self.MAX_FULL_TOKEN_SEQ=max_full_token_seq
        self.SENTENCE_MERGING=sentence_merging

        self.cache_name='cord19_quest_'+\
            basename(dirname(self.CORD_DATA_PATH))+\
            ('_merge' if self.SENTENCE_MERGING else '_nomerge')+\
            '.cache'

        if exists('cache/'+self.cache_name):
            print('SynergyDataSterilizer: using cache')
            self.feedback,self.cord19_processed=pickle.load(open("cache/"+self.cache_name,"rb"))
        else:
            self.grind()

    #Generator that returns questions in the following shape:
    #{'query_id':str, 'query_text':str, 'doc_pos_id': list<str>, 'doc_neg_id': list<str>}
    def questions(self):
        #This generator acts as a translator between our internal structure, and the interface structure
        #Like everything else in this class, can be in a more efficient manner
        for q in self.feedback:
            yield {
                'query_id': '_'.join(self.feedback[q]['qid_set']),
                'query_text': q,
                'doc_pos_id': [x[0] for x in self.feedback[q]['docid_set'] if x[1]],
                'doc_neg_id': [x[0] for x in self.feedback[q]['docid_set'] if not x[1]]}

    #Returns the CORD19 collection dict, no translation needed.
    def collection(self):
        return self.cord19_processed
    
    #Should be more than one function. No time to change. Too bad!
    def grind(self):
        
        #Load feedback files
        feedback_filenames=glob.glob(self.SYNERGY_DATA_PATH+'*Synergy9*/*feedback*')
        print('Loading the following feedback files (CHECK IF THEY ARE RIGHT):')
        [print(x) for x in feedback_filenames]

        feedback=dict()
        for ff in feedback_filenames:
            ffj=json.load(open(ff,'r'))
            for q in ffj['questions']:
                if self.QUESTION_UPPERCASE_FIRST_LETTER: #Fix casing
                    q['body']=q['body'][0:1].upper()+q['body'][1:]
                if q['body'] not in feedback:
                    feedback[q['body']]=dict()
                    feedback[q['body']]['docid_set']=set()
                    feedback[q['body']]['qid_set']=set() #question id set, questions can be cuplicate...
                for qd in q['documents']:
                    feedback[q['body']]['docid_set'].add((qd['id'],qd['golden']))
                    feedback[q['body']]['qid_set'].add(q['id'])

        if self.QUESTION_UPPERCASE_FIRST_LETTER:
            print('[UPPER FIRST LETTER QUESTION] ',end='')
        print('Feedback files loaded (',
              len(feedback.keys()),
              ' questions, ', 
              reduce(lambda a,b:a+len(b['docid_set']),feedback.values(),0), 
              ' total answers)',
              sep='')

        #################################################

        #Handling inconsistent answers being marked both as golden and not golden
        incon_docs_cnt=0
        incon_quest_set=set()
        for q in feedback:
            hit_true,hit_false=list(),list()
            tmp_docid_set=feedback[q]['docid_set'].copy() #Just to iterate and remove on the original docid_set
            for d in tmp_docid_set:
                if d[1] and d[0] not in hit_false:
                    hit_true.append(d[0])
                elif not d[1] and d[0] not in hit_true:
                    hit_false.append(d[0])
                else:
                    #Handle entries
                    incon_docs_cnt+=1
                    incon_quest_set.add(q)
                    if self.INCONSISTENT_RULE=='to_true' or self.INCONSISTENT_RULE=='to_unk':
                        feedback[q]['docid_set'].remove((d[0],False))
                    if self.INCONSISTENT_RULE=='to_false' or self.INCONSISTENT_RULE=='to_unk':
                        feedback[q]['docid_set'].remove((d[0],True))
                    #If self.INCONSISTENT_RULE=='ignore', dont remove anything

        if self.INCONSISTENT_RULE=='to_true':
            print('Inconsistent answers moved to strong positives',end='')
        elif self.INCONSISTENT_RULE=='to_false':
            print('Inconsistent answers moved to strong negatives',end='')
        elif self.INCONSISTENT_RULE=='to_unk':
            print('Inconsistent answers removed/moved to unknown set',end='')
        else: #elsif self.INCONSISTENT_RULE=='ignored':
            print('Inconsistent answers ignored',end='')
        print(' (',
              incon_docs_cnt,
              ' document entries from ',
              len(incon_quest_set),
              ' questions)',
              sep='')

        print('Feedback data current status (',
              len(feedback.keys()),
              ' questions, ', 
              reduce(lambda a,b:a+len(b['docid_set']),feedback.values(),0), 
              ' total answers)',
              sep='')

        #################################################

        #TODO

        #query augumentation with [MASK]s
        #other stuff forgotten along the way

        #################################################

        #Load CORD dataset
        print('Loading data from CORD19 dataset CSV (some filtering already done, but might take a while)')
        cord19=dict()
        cord19_good_cnt,cord19_incomplete_cnt,cord19_notenglish_cnt,cord19_short_cnt=0,0,0,0
        #if exists('cord19.cache'):
        #    print('Loading cached version of CORD19 dataset (if you run into troubles remove cache file cord19.cache to force reimport)')
        #    (cord19,cord19_good_cnt,
        #    cord19_incomplete_cnt,
        #    cord19_notenglish_cnt,
        #    cord19_short_cnt)=pickle.load(open("cord19.cache","rb"))
        #else:
        lang_model = fasttext.load_model('lid.176.bin')
        for r in csv.DictReader(open(self.CORD_DATA_PATH,'r')):
            #Has to have title and abstract and has to have at least one of these 3 IDs
            if (len(r['title'])==0 or len(r['abstract'])==0 or
                not (len(r['pubmed_id'])>0 or len(r['pmcid'])>0 or len(r['arxiv_id'])>0)):
                cord19_incomplete_cnt+=1
            #Abstract has to have more than 5 words
            elif len(r['abstract'].split(' '))<=5: #TODO horrible. No time. Too bad!
                cord19_short_cnt+=1
            #Has to be in english
            elif lang_model.predict(r['abstract'], k=1)[0][0] != '__label__en':
                cord19_notenglish_cnt+=1
            else:
                cord19_good_cnt+=1
                cord19[r['cord_uid']]={'title':r['title'],'abstract':r['abstract']}
        #pickle.dump((cord19,cord19_good_cnt,
        #             cord19_incomplete_cnt,
        #             cord19_notenglish_cnt,
        #             cord19_short_cnt),open("cord19.cache","wb"))

        print('Loaded ',cord19_good_cnt,
              ' valid entries from CORD19 dataset of ',
              cord19_good_cnt+cord19_incomplete_cnt,
              ' total entries (',cord19_incomplete_cnt,
              ' incomplete, ',cord19_notenglish_cnt,
              ' not english, ',cord19_short_cnt,
              ' abstract less than 5 words)',sep='')

    #################################################

        #Document sentence splitting of CORD19 dataset (paralelzed)
        cord19_processed=dict()
        if exists('cord19_processed.cache'):
            print('Sentence splitting CORD19 using cached version (if you run into troubles remove cache file cord19_processed.cache to force reimport)')
            cord19_processed,feedback=pickle.load(open("cord19_processed.cache","rb"))
        else:

            print('Sentence splitting CORD19 using',self.POOL_THREAD_COUNT,'workers (WARNING: takes a LOT of ram and time!)')
            def pool_job(pool_queue,cord19_entries):
                nlp = spacy.load('en_core_web_lg')
                for d in cord19_entries:
                    nlp_out=nlp(cord19_entries[d]['abstract']).sents
                    pool_queue.put((d,[cord19_entries[d]['title']]+[str(i) for i in nlp_out]))

            cord19_keys=list(cord19.keys())
            pool_proc=list()
            pool_queue = Queue()
            pool_jump=math.ceil(len(cord19_keys)/self.POOL_THREAD_COUNT)
            for i in range(0,len(cord19_keys),pool_jump):
                p=Process(target=pool_job,args=(pool_queue,{x:cord19[x] for x in cord19_keys[i:i+pool_jump]}))
                p.start()
                pool_proc.append(p)

            import time
            qsize=pool_queue.qsize()
            while qsize!=len(cord19_keys):
                time.sleep(0.1) #This is a horrible concurrent design pattern. No time to fix. Too bad!
                qsize=pool_queue.qsize()
                print('\rSplitting processing status: ',qsize,'/',len(cord19_keys),
                      '  ',int(qsize/len(cord19_keys)*100),'%',sep='',end='',flush=True)
            print()

            #Join all processes before proceeding
            #TODO: For some reaason the processes are ending but not responding to join. No time to fix. Too bad!
            #for p in pool_proc:
            #    print('AAAAAAAAAAA')
            #    p.join()
            #print('Joined all workers')

            for _ in range(len(cord19_keys)):
                entry=pool_queue.get()
                cord19_processed[entry[0]]=entry[1]

            pickle.dump((cord19_processed,feedback),open("cord19_processed.cache","wb"))
            print('Merged workers data with main thread')

            assert len(cord19_processed)==len(cord19),str(len(cord19_processed))+'!='+str(len(cord19))
            del cord19

        #################################################

        #Remove chinese characters
        chinese_cnt=0
        chinese_re=r'[\u4e00-\u9fff]+'
        for doc in cord19_processed:
            new_sents=list()
            for idx in range(len(cord19_processed[doc])):
                if re.search(chinese_re, cord19_processed[doc][idx])!=None:
                    cord19_processed[doc][idx]=re.sub(chinese_re,'',cord19_processed[doc][idx])
                    chinese_cnt+=1

        print('Sentence cleaning (remove chinese characters) ',
              chinese_cnt,' sentences affected',sep='')

        #################################################

        #Remove website URLs
        url_cnt=0
        url_re=r'(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-z]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        for doc in cord19_processed:
            new_sents=list()
            for idx in range(len(cord19_processed[doc])):
                if re.search(url_re,cord19_processed[doc][idx])!=None:
                    #print(doc)
                    #print(cord19_processed[doc][idx])
                    #print(re.findall(url_re,cord19_processed[doc][idx]))
                    #print(re.sub(url_re,'',cord19_processed[doc][idx]))
                    #print()
                    cord19_processed[doc][idx]=re.sub(url_re,'',cord19_processed[doc][idx])
                    url_cnt+=1

        print('Sentence cleaning (remove URLs) ',
              url_cnt,' sentences affected',sep='')
        
        #################################################

        #Merge sentences that have less than N words
        if self.SENTENCE_MERGING:
            old_sent_cnt,new_sent_cnt=0,0
            for doc in cord19_processed:
                new_sents=list()
                sent_buf=str()
                sent_buf_size=0
                for sent in cord19_processed[doc]:
                    old_sent_cnt+=1
                    sent_len=len(sent.split(' '))
                    if sent_len>self.MAX_FULL_TOKEN_SEQ:
                        #If sentence too large, flush buffer and flush sentence
                        new_sents.append(sent_buf)
                        sent_buf=str()
                        sent_buf_size=0
                        new_sents.append(sent)
                        new_sent_cnt+=1
                    elif sent_buf_size+sent_len>self.MAX_FULL_TOKEN_SEQ:
                        #If sentence would overflow buffer, flush buffer add sentence to buffer
                        new_sents.append(sent_buf)
                        new_sent_cnt+=1

                        sent_buf=sent
                        sent_buf_size=sent_len
                    else:
                        #Simply add sentence to buffer
                        if len(sent_buf)==0:
                            sent_buf+=sent
                        else:
                            sent_buf+=' '+sent
                        sent_buf_size+=sent_len
                if sent_buf_size>0: #Flush the buffer one last time
                    new_sents.append(sent_buf)
                    new_sent_cnt+=1
                cord19_processed[doc]=new_sents

            print('Sentence merging (for '
                  ,self.MAX_FULL_TOKEN_SEQ,' words) complete ',
                  old_sent_cnt,' sentences condensed in ',
                  new_sent_cnt,' sentences (',
                  int(old_sent_cnt/new_sent_cnt*100),'\% compression rate)'
                  ,sep='')
        else:
            print('NO sentence merging')

        #################################################

        #Remove revoked articles referenced on Synergy (from feedback)
        revoked_doc_cnt=0
        revoked_qdoc_set=set()
        for q in feedback:
            tmp_docid_set=feedback[q]['docid_set'].copy() #Just to iterate and remove on the original docid_set
            for doc in tmp_docid_set:
                if doc[0] not in cord19_processed:
                    feedback[q]['docid_set'].remove(doc)
                    revoked_doc_cnt+=1
                    revoked_qdoc_set.add(q)

        print('Removed ', revoked_doc_cnt,' revoked CORD19 documents from ',len(revoked_qdoc_set),' questions on feedback data',sep='')

        print('Feedback data current status (',
              len(feedback.keys()),
              ' questions, ', 
              reduce(lambda a,b:a+len(b['docid_set']),feedback.values(),0), 
              ' total answers)',
              sep='')

        #################################################

        #Remove questions without answers or with only strong negative answers
        deleted_quest_cnt=0
        for q in list(feedback.keys()):
            #If it has at least one strong positive, keep. Otherwise delete
            if not reduce(lambda a,b:a or b[1],feedback[q]['docid_set'], False): #TODO check logic
                deleted_quest_cnt+=1
                del feedback[q]

        print('Removed questions with no document answers (',
              deleted_quest_cnt,
              ' empty questions)',
              sep='')

        print('Feedback data current status (',
              len(feedback.keys()),
              ' questions, ', 
              reduce(lambda a,b:a+len(b['docid_set']),feedback.values(),0), 
              ' total answers)',
              sep='')

        #################################################
        
        #Make output structures accessible
        self.feedback=feedback
        self.cord19_processed=cord19_processed
        
        pickle.dump((feedback,cord19_processed),open("cache/"+self.cache_name,"wb"))
        
@click.command()
@click.option('--synergy_data_path', '-s', default='../data/synergy_datasety/' ,help='Path to Synergy data')
@click.option('--cord_data_path', '-c', default='../data/cord-19/2022-01-31/metadata.csv' ,help='Path to cord data')
@click.option('--question_uppercase_first_letter', '-q', is_flag=True, help='First letter uppercase')
@click.option('--inconsistent_rule', '-i', default='to_false', help='Inconsistent Rule')
@click.option('--max_full_token_seq', '-m',  default=80 ,help='Max Full token Sequence', type=int)
@click.option('--pool_thread_count','-p' , default=25, help='Pool thread count', type=int)
@click.option('--sentence_merging','-a' , is_flag=True, help='Activate sentence merging')
def data(synergy_data_path,
         cord_data_path,
         question_uppercase_first_letter,
         inconsistent_rule,
         max_full_token_seq,
         pool_thread_count,
         sentence_merging):

    to_print=False
    
    sds=SynergyDataSterilizer(
                 synergy_data_path,
                 cord_data_path,
                 question_uppercase_first_letter,
                 inconsistent_rule,
                 max_full_token_seq,
                 pool_thread_count,
                 sentence_merging)
    
    questions=sds.questions()
    if to_print:
        print(next(questions))
    
    collection=sds.collection()
    c=list(collection.keys())[0]
    if to_print:
        print(c,collection[c])
        
if __name__ == "__main__":
    data()
