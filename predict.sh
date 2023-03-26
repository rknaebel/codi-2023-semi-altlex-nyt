#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0
LIMIT=0

for corpus in pdtb3 nyt
do
    python3 predict_labels.py ${corpus} "models/ensemble_1/model_*" -r -o results/v4/m1/${corpus}.json
    python3 predict_labels.py ${corpus} "models/ensemble_2/model_*" -r -o results/v4/m2/${corpus}.json
    break
done
