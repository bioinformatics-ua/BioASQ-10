# -*- coding: utf-8 -*-
"""
Code from Anserini, fusion.py

Adapted by: Tiago Almeida

Anserini: A Lucene toolkit for replicable information retrieval research
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import enum

from typing import List
from trectools import fusion, TrecRun
import bm25_search
import json

class FusionMethod(enum.Enum):
    RRF = 'RRF'
    COMBO_SUM = 'COMBO_SUM'


def load_trec_runs(paths: List[str]) -> List[TrecRun]:
    print(f'Loading {len(paths)} runs')
    return [TrecRun(path) for path in paths]


def perform_fusion(method: FusionMethod, runs: List[TrecRun], output_path: str, max_docs: int) -> None:
    print('Performing fusion ->', method)

    if method == FusionMethod.RRF:
        fused_run = fusion.reciprocal_rank_fusion(runs, max_docs=max_docs)
        fused_run.print_subset(output_path, topics=fused_run.topics())
    elif method == FusionMethod.COMBO_SUM:
        with open(output_path, 'w+') as f:
            fusion.combos(runs, strategy="sum", max_docs=max_docs, output=f)
    else:
        raise Exception(f'The requested method {method} is not implemented.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='performs various methods of fusion supported by trectools')
    parser.add_argument('--runs', type=str, nargs='+',
                        default=[], required=True, help='a list of run files')
    parser.add_argument('--out', type=str,
                        default="fused.txt", required=False, help='the output path of the fused run')
    parser.add_argument('--max_docs', type=int,
                        default=1000, required=False, help='maximum of hits')
    parser.add_argument('--q_bio', default=None, type=str, help="Queries in the BioASQ format")
    parser.add_argument('--rf', default=None, type=str, help="relevance feedback file")

    args = parser.parse_args()

    trec_runs = load_trec_runs(args.runs)
    perform_fusion(FusionMethod.RRF, trec_runs, args.out, args.max_docs)

    print(f'Fusion successful -> {args.out}')
    
    bioasq_run = bm25_search.load_trec_run_to_bioasq(args.out, args.rf)
    
    if args.q_bio is not None:
        
        with open(args.q_bio, "r") as f:
            bioasq_queries = json.load(f)["questions"]
            
        output_file_json = args.out.split(".")[0] + ".json"
        
        bm25_search.write_as_bioasq(bioasq_queries, bioasq_run, output_file_json)