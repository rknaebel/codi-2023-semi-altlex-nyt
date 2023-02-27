#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=1
MODEL_PATH=models

for i in {1..3}
do
    for j in {1..3}
    do
        python3 train_label.py pdtb3 altlex -b 12 --split-ratio 0.9 --save-path $MODEL_PATH/ensemble_$i/model_$j/ --test-set
        python3 train_sense.py pdtb3 altlex -b 16 --split-ratio 0.9 --save-path $MODEL_PATH/ensemble_$i/model_$j/ --test-set
    done
done