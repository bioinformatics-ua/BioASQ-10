#!/bin/bash

set -e

# run the python scripts in order to recreate the runs

## NOTE that there were a bug in rnd3 which is now fixed so, the produced runs differ from the submitted ones.


echo "Preparing the CORD-19 collection for each of the rounds, by default the script will use the available cache files"
## rnd 1 uncomment this line to preprocess the rnd 1, however be aware that will take a lot of time
#python data_sterilizer.py --cord_data_path ..data/cord-19/2021-11-15/metadata.csv
## rnd 2 uncomment this line to preprocess the rnd 1, however be aware that will take a lot of time
#python data_sterilizer.py --cord_data_path ..data/cord-19/2021-12-20/metadata.csv
## rnd 3 
python data_sterilizer.py --cord_data_path ..data/cord-19/2022-01-03/metadata.csv
## rnd 4
python data_sterilizer.py --cord_data_path ..data/cord-19/2022-01-31/metadata.csv

echo "Indexing using pyserini"
# by default only rnd 3 and 4 will be indexed. This is also already available in the cache folder
python pyserini_indexer.py

echo "Preparing the dataset (synergy testset and feedback data)"
python process_questions_feedback_data.py

echo "Building the runs for rnd 3:"
echo -e "\t rnd3/bioinfo-0"
python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset3_trecformat.tsv cache/indexes/rnd3-doc-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round3_trecformat.txt -q_bio data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset3 -k1 1.8 -b 0.8 -fb_terms 89 -fb_docs 1 -qw 0.6 -name "rnd3/bioinfo-0"

echo -e "\t rnd3/bioinfo-1"
python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset3_trecformat.tsv cache/indexes/rnd3-doc-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round3_trecformat.txt -k1 1.8 -b 0.8 -fb_terms 89 -fb_docs 1 -qw 0.6 -name "rnd3/bioinfo-1-0"

python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset3_trecformat.tsv cache/indexes/rnd3-doc-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round3_trecformat.txt -k1 1.8 -b 0.8 -fb_terms 40 -fb_docs 1 -qw 0.6 -name "rnd3/bioinfo-1-1"

python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset3_trecformat.tsv cache/indexes/rnd3-doc-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round3_trecformat.txt -k1 1.8 -b 0.8 -fb_terms 88 -fb_docs 6 -qw 0.55 -name "rnd3/bioinfo-1-2"

python rrfusion.py --runs runs/rnd3/bioinfo-1-0.txt runs/rnd3/bioinfo-1-1.txt runs/rnd3/bioinfo-1-2.txt --out runs/rnd3/bioinfo-1.txt --rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round3_trecformat.txt --q_bio data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset3

echo -e "\t rnd3/bioinfo-2"

python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset3_trecformat.tsv cache/indexes/rnd3-merge-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_snippets_feedback_round3_merge_trecformat.txt -q_bio data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset3 -k1 0.7 -b 0.4 -fb_terms 68 -fb_docs 10 -qw 0.5 -d_rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round3_trecformat.txt -name "rnd3/bioinfo-2"

echo -e "\t rnd3/bioinfo-3"

python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset3_trecformat.tsv cache/indexes/rnd3-merge-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_snippets_feedback_round3_merge_trecformat.txt -k1 0.7 -b 0.4 -fb_terms 68 -fb_docs 10 -qw 0.5 -d_rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round3_trecformat.txt -name "rnd3/bioinfo-3-0"

python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset3_trecformat.tsv cache/indexes/rnd3-merge-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_snippets_feedback_round3_merge_trecformat.txt -k1 0.7 -b 0.4 -fb_terms 68 -fb_docs 10 -qw 0.5 -d_rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round3_trecformat.txt -name "rnd3/bioinfo-3-1"

python rrfusion.py --runs runs/rnd3/bioinfo-3-0.txt runs/rnd3/bioinfo-3-1.txt --out runs/rnd3/bioinfo-3.txt --rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round3_trecformat.txt --q_bio data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset3

echo "Building the runs for rnd 4:"
echo -e "\t rnd4/bioinfo-0"
python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset4_trecformat.tsv cache/indexes/rnd4-doc-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt -q_bio data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset4 -k1 0.6 -b 0.8 -fb_terms 96 -fb_docs 9 -qw 0.5 -name "rnd4/bioinfo-0"

echo -e "\t rnd4/bioinfo-1"
python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset4_trecformat.tsv cache/indexes/rnd4-doc-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt -k1 0.6 -b 0.8 -fb_terms 96 -fb_docs 9 -qw 0.5 -name "rnd4/bioinfo-1-0"

python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset4_trecformat.tsv cache/indexes/rnd4-doc-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt -k1 1.0 -b 0.8 -fb_terms 90 -fb_docs 1 -qw 0.5 -name "rnd4/bioinfo-1-1"

python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset4_trecformat.tsv cache/indexes/rnd4-doc-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt -k1 1.0 -b 0.6 -fb_terms 66 -fb_docs 1 -qw 0.6 -name "rnd4/bioinfo-1-2"

python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset4_trecformat.tsv cache/indexes/rnd4-doc-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt -k1 1.1 -b 0.6 -fb_terms 62 -fb_docs 10 -qw 0.6 -name "rnd4/bioinfo-1-3"

python rrfusion.py --runs runs/rnd4/bioinfo-1-0.txt runs/rnd4/bioinfo-1-1.txt runs/rnd4/bioinfo-1-2.txt runs/rnd4/bioinfo-1-3.txt --out runs/rnd4/bioinfo-1.txt --rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt --q_bio data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset4

echo -e "\t rnd4/bioinfo-2"
python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset4_trecformat.tsv cache/indexes/rnd4-merge-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_snippets_feedback_round4_merge_trecformat.txt -q_bio data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset4 -k1 0.8 -b 0.8 -fb_terms 62 -fb_docs 5 -qw 0.5 -d_rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt -name "rnd4/bioinfo-2"

echo -e "\t rnd4/bioinfo-3"
python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset4_trecformat.tsv cache/indexes/rnd4-merge-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_snippets_feedback_round4_merge_trecformat.txt -k1 0.8 -b 0.8 -fb_terms 62 -fb_docs 5 -qw 0.5 -d_rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt -name "rnd4/bioinfo-3-0"

python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset4_trecformat.tsv cache/indexes/rnd4-merge-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_snippets_feedback_round4_merge_trecformat.txt -k1 0.8 -b 0.8 -fb_terms 55 -fb_docs 5 -qw 0.5 -d_rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt -name "rnd4/bioinfo-3-1"

python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset4_trecformat.tsv cache/indexes/rnd4-merge-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_snippets_feedback_round4_merge_trecformat.txt -k1 0.9 -b 0.8 -fb_terms 60 -fb_docs 5 -qw 0.5 -d_rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt -name "rnd4/bioinfo-3-2"

python bm25_search.py data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_testset4_trecformat.tsv cache/indexes/rnd4-merge-cord-collection/ -rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_snippets_feedback_round4_merge_trecformat.txt -k1 0.6 -b 0.6 -fb_terms 43 -fb_docs 4 -qw 0.5 -d_rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt -name "rnd4/bioinfo-3-3"

python rrfusion.py --runs runs/rnd4/bioinfo-3-0.txt runs/rnd4/bioinfo-3-1.txt runs/rnd4/bioinfo-3-2.txt runs/rnd4/bioinfo-3-3.txt --out runs/rnd4/bioinfo-3.txt --rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt --q_bio data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset4

echo -e "\t rnd4/bioinfo-4"
#python rrfusion.py --runs runs/rnd4/bioinfo-1.txt runs/rnd4/bioinfo-3.txt --out runs/rnd4/bioinfo-4.txt --rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt --q_bio data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset4

python rrfusion.py --runs runs/rnd4/bioinfo-1-0.txt runs/rnd4/bioinfo-1-1.txt runs/rnd4/bioinfo-1-2.txt runs/rnd4/bioinfo-1-3.txt runs/rnd4/bioinfo-3-0.txt runs/rnd4/bioinfo-3-1.txt runs/rnd4/bioinfo-3-2.txt runs/rnd4/bioinfo-3-3.txt --out runs/rnd4/bioinfo-4.txt --rf data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_2022_documents_feedback_round4_trecformat.txt --q_bio data/dataset/BioASQ_Synergy10/BioASQ-taskSynergy_v2022-testset4

rm runs/rnd3/*.txt
rm runs/rnd4/*.txt

echo "Final runs written to runs/rnd3 and runs/rnd4"
