#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1
export CUDA_LAUNCH_BLOCKING=1
model_path=models/self-ensemble-4

for i in {1..5}
do
    python3 train_label.py pdtb3 -b 32 --save-path $model_path/final/base/m${i} \
                                 --valid-seed $i \
                                 --val-metric f1-score \
                                 --test-set
done

python3 test_label.py pdtb3 -b 32 --save-path "$model_path/final/base/m*" --mode average