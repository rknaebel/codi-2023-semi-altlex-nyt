#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1
model_path=$2


python3 train_label.py pdtb3 -b 32 --save-path $model_path/v1/m1 --test-set --weighted-loss --valid-seed 1 --val-metric f1-score

for i in {1..7}
do
    python3 predict_labels.py pdtb3 "$model_path/v${i}/m1" -o $model_path/v${i}/m1/pdtb3.json
    python3 train_sense.py pdtb3 $model_path/v${i}/m1/pdtb3.json -b 32 --save-path $model_path/v$i/m1 --hidden 256,64
    python3 predict_signals.py pdtb3 "$model_path/v${i}/m1" -o $model_path/v${i}/m1/pdtb3.full.json --is-relation-threshold 0.33
    python3 predict_signals.py nyt "$model_path/v${i}/m1" -o $model_path/v${i}/m1/nyt.json --sample-ratio 0.3 --limit 5000 --is-relation-threshold 0.33

    if [[ $i -lt 7 ]]; then
        python3 train_label.py pdtb3 --corpus-plus $model_path/v${i}/m1/nyt.json -b 32 \
                                     --save-path "$model_path/v$((i+1))/m1" \
                                     --continue-model "$model_path/v$i/m1" \
                                     --pseudo-limit-ratio 0.70 \
                                     --num-epochs 30 \
                                     --initial-learning-rate 1e-6 \
                                     --valid-seed 1 \
                                     --sort-select \
                                     --val-metric recall \
                                     --test-set --weighted-loss
    fi
done

python3 predict_signals.py nyt $model_path/v7/m1 -o $model_path/v7/nyt.full.json --sample-ratio 0.5 --limit 10000 --is-relation-threshold 0.80

for i in {1..3}
do
    python3 train_label.py pdtb3 --corpus-plus $model_path/v7/nyt.full.json -b 32 \
                                 --save-path $model_path/final/m${i} \
                                 --valid-seed $i \
                                 --num-epochs 50 \
                                 --test-set
done

python3 test_label.py pdtb3 -b 32 --save-path "$model_path/final/m*"