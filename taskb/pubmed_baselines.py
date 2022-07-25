from urllib.request import urlopen
from multiprocessing import Process,Manager
from lxml import etree
import os
import re
import gzip
import csv
import json
import math
import queue
import time
import random
from itertools import chain
from pathlib import Path

first_year=2013
last_year=2022
proc_cnt=15
basepath=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data','pubmed')

def get_pak_links(year):
    url='https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/'+str(year)+'.html'
    output=urlopen(url).read()
    tree=etree.fromstring(output.decode('utf-8'),etree.HTMLParser())

    ret=list()
    for i in tree.xpath('/html/body/ul[2]/li/a'):
        if i.text!='gzip_all':
            ret.append(i.get('href')) 
    random.shuffle(ret)
    return ret

def dissect_xml(tree):
    articles=tree.findall('//MedlineCitation')
    l=[]

    for article in articles:
        pmid = article.find("PMID").text
        try:
            title = ''.join(list(article.find("Article/ArticleTitle").itertext()))
        except:
            continue

        category = 'Label' #"NlmCategory" if nlm_category else "Label"
        if article.find("Article/Abstract/AbstractText") is not None:
            # parsing structured abstract
            if len(article.findall("Article/Abstract/AbstractText")) > 1:
                 abstract_list = list()
                 for abstract in article.findall("Article/Abstract/AbstractText"):
                    section = abstract.attrib.get(category, "")
                    if section != "UNASSIGNED":
                        abstract_list.append("\n")
                        abstract_list.append(abstract.attrib.get(category, ""))
                    section_text = stringify_children(abstract).strip()
                    abstract_list.append(section_text)
                 abstract = "\n".join(abstract_list).strip()
            else:
                abstract = (
                    stringify_children(article.find("Article/Abstract/AbstractText")).strip() or ""
                )
        elif article.find("Article/Abstract") is not None:
            abstract = stringify_children(article.find("Article/Abstract")).strip() or ""
        else:
            continue
        l.append({"pmid": pmid, "title": title, "abstract": abstract})

    #    try:
    #        title = ''.join(list(article.find("Article/ArticleTitle").itertext()))
    #        #abstract = ''.join(list(article.find("Article/Abstract/AbstractText").itertext()))
    #        abstract = ' '.join([''.join(x.itertext()) for x in article.findall("Article/Abstract/AbstractText")])
    #        if len(abstract)==0:
    #            raise Exception("Empty abstract")
    #        #if pmid=='19724244':
    #        #    print(etree.tostring(article.find("Article"),pretty_print=True).decode())
    #        #    print(abstract)
    return l

def stringify_children(node):
    """
    Filters and removes possible Nones in texts and tails
    ref: http://stackoverflow.com/questions/4624062/get-all-text-inside-a-tag-in-lxml
    """
    parts = (
        [node.text]
        + list(chain(*([c.text, c.tail] for c in node.getchildren())))
        + [node.tail]
    )
    return "".join(filter(None, parts))


def handle_pak(links,queue,fax_id,fax_queue):
    for link in links:
        while True:
            fax_queue.put((fax_id,'.'))
            try:
                xmlgz_fd=urlopen(link)
                break
            except:
                fax_queue.put((fax_id,'E'))
                time.sleep(1)

        xml_fd=gzip.open(xmlgz_fd)
        fax_queue.put((fax_id,':'))
        xml_tree=etree.parse(xml_fd)
        fax_queue.put((fax_id,'!'))
        to_append=dissect_xml(xml_tree)
        queue.put(to_append) 

        del to_append
        del xml_tree
        del xml_fd
        del xmlgz_fd

def fax_machine(fax_queue,proc_cnt,pak_cnt,pool_queue):
    current_fax=['0']*proc_cnt
    current_cnt=0
    print('0'*proc_cnt,end='',flush=True)
    while current_cnt!=pak_cnt:
        fax_update=fax_queue.get()
        if fax_update[1]=='!':
            current_cnt+=1
        current_fax[fax_update[0]]=fax_update[1]
        print('\b'*300,
                ''.join(current_fax),
                ' ',current_cnt,
                '/',pak_cnt,
                ' (write buf size: ',pool_queue.qsize(),')',
                sep='',end='',flush=True)
    print()

def data_destilator():
    links=dict()
    for i in range(first_year,last_year+1):
        pl=get_pak_links(i)
        links[i]=pl
        print(i,'-',len(pl))

    for year in range(first_year,last_year+1):
        print('Fetching',year)

        filename=os.path.join(basepath,'pubmed'+str(year)+'.jsonl')
        #out_csv=gzip.open(filename, 'wt', newline='') 
        out_json=open(filename, 'w', newline='\n') 
        #csv_writer=csv.DictWriter(out_csv,fieldnames=['pmid','title','abstract'])
        #csv_writer.writeheader()

        pool_jump=math.ceil(len(links[year])/proc_cnt)
        pool_proc=list()
        man=Manager()
        pool_queue=man.Queue()
        fax_queue=man.Queue()

        p=Process(target=fax_machine,args=(fax_queue,proc_cnt,len(links[year]),pool_queue))
        p.start()

        for i,n in enumerate(range(0,len(links[year]),pool_jump)):
            pak_slice=links[year][n:(n+pool_jump)]
            p=Process(target=handle_pak,args=(pak_slice,pool_queue,i,fax_queue))
            p.start()
            pool_proc.append(p)

        progress=0
        #to_write=list()
        while progress!=len(links[year]):
            pq=pool_queue.get()
            for p in pq:
                out_json.write(json.dumps(p)+'\n')
            del pq
            progress+=1

        #json.dump(to_write,out_csv)
        #out_csv.write(json.dumps(to_write))
        #csv_writer.close()
        out_json.close()
        time.sleep(0.5)

#############################

def baseline_gen():

    #Load first year
    cbl=dict()
    cnt=0

    #Load all the years from the most recent
    for year in reversed(range(first_year,last_year+1)):
        print('----',year,'----')
        with open(os.path.join(basepath,f'pubmed{year}.jsonl'),'r') as fp:
            l=fp.readline()
            cnt=0
            while l!=None and len(l)!=0:
                cnt+=1
                if cnt%10000==0:
                    print('>>>',cnt,end='\r',flush=True)
                c=json.loads(l)
                if c['pmid'] in cbl: #only add year
                    cbl[c['pmid']]['years'][0:0]=[year]
                else:
                    cbl[c['pmid']]=c
                    c['years']=[year]
                    c['contents']=c['title']+'. '+c['abstract']
                    c['id']=c['pmid']
                    del c['pmid']
                    #del c['title']
                    del c['abstract']
                del l
                l=fp.readline()

    #Dump entries
    out=open(os.path.join(basepath,'pubmed_all.jsonl'),'w')
    for pmid in cbl:
        out.write(json.dumps(cbl[pmid])+'\n')
    out.close()

#############################

def rm_temporary():
	for year in range(first_year,last_year+1):
		Path(os.path.join(basepath,f'pubmed{year}.jsonl')).unlink()
#############################

def main():
	data_destilator()
	baseline_gen()
	rm_temporary()

main()
