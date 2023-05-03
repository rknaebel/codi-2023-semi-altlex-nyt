#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1
model_path=models/final/senses/no-sampling

for i in {1..10}; do
    python3 train_sense.py pdtb3 models/self-ensemble-4/v1/pdtb3.json -b 32 --save-path $model_path/m$i --hidden 256,64 --random-seed $i
done

for i in {1..10}; do
    python3 test_sense.py pdtb3 models/self-ensemble-4/v1/pdtb3.json -b 32 --save-path $model_path/m$i --random-seed $i
done
