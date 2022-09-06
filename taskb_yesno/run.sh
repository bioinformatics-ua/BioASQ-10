#!/bin/bash

set -e

echo "Prepare dataset"

python cli_prepare_dataset_phase_B.py --split_perc 0.98 --ds 1 --dl_suffix para_quest_sent

echo "Train non ensemble models"
# System 1
python cli_train --name system1 --lr 0.001 --batch_size 64 --epoch 15 --bert_trainable_cnt 0 --dense_layers 768 --dropout 0.2

#System 2
python cli_train --name system2 --lr 0.0001 --batch_size 64 --epoch 10 --bert_trainable_cnt 5 --dense_layers 768 --dropout 0.4

#System 3
python cli_train --name system3 --lr 0.0005 --batch_size 64 --epoch 15 --bert_trainable_cnt 0 --dense_layers 192 --dropout 0.4

#System 4
python cli_train --name system4 --lr 0.0001 --batch_size 64 --epoch 10 --bert_trainable_cnt 7 --dense_layers 768 --dropout 0.4


echo "Generate test results for individual models"

python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system1.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset3.json --save_path test_results/ts3
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system1.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset4.json --save_path test_results/ts4
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system1.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset5.json --save_path test_results/ts5
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system1.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset6.json --save_path test_results/ts6

python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system2.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset3.json --save_path test_results/ts3
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system2.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset4.json --save_path test_results/ts4
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system2.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset5.json --save_path test_results/ts5
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system2.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset6.json --save_path test_results/ts6

python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system3.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset3.json --save_path test_results/ts3
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system3.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset4.json --save_path test_results/ts4
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system3.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset5.json --save_path test_results/ts5
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system3.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset6.json --save_path test_results/ts6

python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system4.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset3.json --save_path test_results/ts3
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system4.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset4.json --save_path test_results/ts4
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system4.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset5.json --save_path test_results/ts5
python cli_get_test_results.py --model_path ../taskb/cache/saved_models/system4.cfg --testset_path test_sets/BioASQ-task10bPhaseB-testset6.json --save_path test_results/ts6

echo "Generate ensemble model"

python cli_ensemble_of_runs.py --runs test_results/ts3/*.json  --testset_path test_sets/BioASQ-task10bPhaseB-testset3.json --save_path test_results/ts3

python cli_ensemble_of_runs.py --runs test_results/ts4/*.json  --testset_path test_sets/BioASQ-task10bPhaseB-testset4.json --save_path test_results/ts4

python cli_ensemble_of_runs.py --runs test_results/ts5/*.json  --testset_path test_sets/BioASQ-task10bPhaseB-testset5.json --save_path test_results/ts5

python cli_ensemble_of_runs.py --runs test_results/ts6/*.json  --testset_path test_sets/BioASQ-task10bPhaseB-testset6.json --save_path test_results/ts6
