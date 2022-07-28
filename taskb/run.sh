#!/bin/bash

set -e

# run the python scripts in order to recreate the runs

## NOTE that there were a bug in rnd3 which is now fixed so, the produced runs differ from the submitted ones.


echo "BM25 runs"
python bm25_inference.py data/dataset/BioASQ-task10bPhaseA-testset1 runs/batch1/baseline_bm25.json
python bm25_inference.py data/dataset/BioASQ-task10bPhaseA-testset2 runs/batch2/baseline_bm25.json
python bm25_inference.py data/dataset/BioASQ-task10bPhaseA-testset3 runs/batch3/baseline_bm25.json
python bm25_inference.py data/dataset/BioASQ-task10bPhaseA-testset4 runs/batch4/baseline_bm25.json
python bm25_inference.py data/dataset/BioASQ-task10bPhaseA-testset5 runs/batch5/baseline_bm25.json
python bm25_inference.py data/dataset/BioASQ-task10bPhaseA-testset6 runs/batch6/baseline_bm25.json


echo "Parade runs"
CUDA_VISIBLE_DEVICES="0" POLUS_JIT=False python parade_inference.py cache/saved_models/pleasant-deluge-1.cfg data/dataset/BioASQ-task10bPhaseA-testset5 runs/batch5/baseline_bm25.json -o runs/batch5


echo "T-UPWM runs"
cd local-T-UPWM

CUDA_VISIBLE_DEVICES="1" python create_bioasq_run.py ../data/dataset/BioASQ-task10bPhaseA-testset1 ../runs/batch1/baseline_bm25.json trained_models/dandy-elevator-14_val_collection0_recall\@10 -o ../runs/batch1/dandy-elevator-14_val_collection0_recall

cd -
