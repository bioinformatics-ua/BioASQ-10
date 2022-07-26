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

