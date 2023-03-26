#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1
model_path=$2

python3 train_label.py pdtb3 -b 32 --save-path $model_path/v1/m1 --test-set --weighted-loss --valid-seed 1
python3 train_label.py pdtb3 -b 32 --save-path $model_path/v1/m2 --test-set --weighted-loss --valid-seed 2

for i in {1..3}
do
    python3 predict_labels.py pdtb3 "$model_path/v${i}/m1" -o $model_path/v${i}/m1/pdtb3.json
    python3 train_sense.py pdtb3 $model_path/v${i}/m1/pdtb3.json -b 32 --save-path $model_path/v$i/m1 --hidden 1024,128
    python3 predict_signals.py nyt "$model_path/v${i}/m1" -o $model_path/v${i}/m1/nyt.json  --sample-ratio 0.3 --limit 5000 --is-relation-threshold 0.4
    python3 predict_signals.py pdtb3 "$model_path/v${i}/m1" -o $model_path/v${i}/m1/pdtb3.full.json --is-relation-threshold 0.4

    python3 predict_labels.py pdtb3 "$model_path/v${i}/m2" -o $model_path/v${i}/m2/pdtb3.json
    python3 train_sense.py pdtb3 $model_path/v${i}/m2/pdtb3.json -b 32 --save-path $model_path/v$i/m2 --hidden 1024,128
    python3 predict_signals.py nyt "$model_path/v${i}/m2" -o $model_path/v${i}/m2/nyt.json  --sample-ratio 0.3 --limit 5000 --is-relation-threshold 0.4
    python3 predict_signals.py pdtb3 "$model_path/v${i}/m2" -o $model_path/v${i}/m2/pdtb3.full.json --is-relation-threshold 0.4

    if [[ $i -lt 3 ]]; then
        python3 train_label.py pdtb3 --corpus-plus $model_path/v${i}/m1/nyt.json -b 32 \
                                     --save-path  "$model_path/v$((i+1))/m2" \
                                     --continue-model "$model_path/v$i/m2" \
                                     --valid-seed 2 \
                                     --sort-select \
                                     --test-set --weighted-loss
        python3 train_label.py pdtb3 --corpus-plus $model_path/v${i}/m2/nyt.json -b 32 \
                                     --save-path  "$model_path/v$((i+1))/m1" \
                                     --continue-model "$model_path/v$i/m1" \
                                     --valid-seed 1 \
                                     --sort-select \
                                     --test-set --weighted-loss
    fi
done


python3 predict_signals.py nyt "$model_path/v3/m[12]" -o $model_path/v3/nyt.full.json --sample-ratio 0.5 --limit 10000  --is-relation-threshold 0.80

for i in {1..5}
do
    python3 train_label.py pdtb3 --corpus-plus $model_path/v3/nyt.full.json -b 32 \
                                 --save-path $model_path/final/m${i} \
                                 --valid-seed $i \
                                 --num-epochs 50 \
                                 --test-set
done

python3 test_label.py pdtb3 -b 32 --save-path 'models/co-train/final/m*'