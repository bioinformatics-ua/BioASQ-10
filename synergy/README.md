# BioASQ-Synergy 10

## Usage

To reproduce our submitted runs you just need to execute the following command.

```
$ ./run.sh
```

This script is already configured to run all of the python scripts in order to produce the round 3 and round 4 runs. Note that, as mentioned in the paper, we had a bug in round 3 so the produced files are the correct ones, which differ from those that we submitted. Nevertheless, the round 4 runs are exactly the same.


## Documentation

### Python scripts

* `data_sterilizer.py`: Preprocesses the CORD-19 collection for each available round. The preprocessing steps are the same as described in the article. By default, we made available a cache file that holds the respective preprocessing of the CORD-19 for rounds 3 and 4. The produced files are written to `cache/`.
* `pyserini_indexer.py`: Uses the pyserini to index the previously preprocessed collection. Similarly to the above script, we made available the indexes for rounds 3 and 4. The produced files are written to data/dataset. The produced files are written to `cache/indexes`.
* `process_questions_feedback_data.py`: Preprocessing of the synergy testset and feedback data. The produced files are written to `data/dataset`.
* `bm25_search.py`: Runs the bm25 algorithm with rm3 feedback. The produced files are written to `runs`.
* `rrfusion.py`: Runs the rank reciprocal fusion algorithm responsible for producing an ensemble of runs. The produced files are written to `runs`.

### Directory structure

```
.
├── cache/
│       Holds intermediate steps from the preprocessing scripts and indexer.
│
├── data/
│       Holds the raw data of the data collections and synergy testsets
│
└── runs/
        Default output folder where the produced runs are stored.
```


