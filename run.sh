#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1
model_path=$2

python3 train_label.py pdtb3 -b 16 --save-path $model_path/v1 --test-set --weighted-loss

#python3 predict_labels.py pdtb3 "$model_path/v1" -r -o $model_path/v1/pdtb3.json
#python3 train_pseudo_sense.py pdtb3 $model_path/v1/pdtb3.json -b 16 --save-path $model_path/v1
#python3 predict_signals.py nyt "$model_path/v1" -r -o $model_path/v1/nyt.json
#python3 train_pseudo_label.py pdtb3 $model_path/v1/nyt.json -b 16 --save-path $model_path/v2 --test-set

for i in {1..5}
do
    python3 predict_labels.py pdtb3 $model_path/v${i} -r -o $model_path/v${i}/pdtb3.json
    python3 train_pseudo_sense.py pdtb3 $model_path/v${i}/pdtb3.json -b 16 --save-path $model_path/v${i}
    python3 predict_signals.py nyt $model_path/v${i} -r -o $model_path/v${i}/nyt.json
    python3 train_pseudo_label.py pdtb3 $model_path/v${i}/nyt.json -b 16 --save-path $model_path/v$((i+1)) --test-set --weighted-loss
done

for i in {1..5}
do
    python3 train_pseudo_label.py pdtb3 $model_path/v5/nyt.json -b 16 --save-path $model_path/final/${i} --test-set
done
