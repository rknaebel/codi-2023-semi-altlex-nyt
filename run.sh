#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0
LIMIT=0

for corpus in pdtb3 essay bbc unsc nyt aes anthology ted
do
    corpus=nyt
#    python3 predict_signals.py ${corpus} "models/ensemble_1/model_*" -r --limit $LIMIT -o results/m1/${corpus}.csv
    python3 predict_signals.py ${corpus} "models/ensemble_2/model_*" -r --limit $LIMIT -o results/m2/${corpus}.csv
    python3 predict_signals.py ${corpus} "models/ensemble_3/model_*" -r --limit $LIMIT -o results/m3/${corpus}.csv
    break
done
