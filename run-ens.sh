#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1
export CUDA_LAUNCH_BLOCKING=1
model_path=models/self-ensemble-4

#for i in {1..5}
#do
#    python3 train_label.py pdtb3 -b 32 --save-path $model_path/base/m${i} \
#                                 --valid-seed $i \
#                                 --num-epochs 30 \
#                                 --val-metric f1-score \
#                                 --test-set
#    python3 test_label.py pdtb3 -b 32 --save-path $model_path/base/m${i}
#done
#
#python3 test_label.py pdtb3 -b 32 --save-path "$model_path/base/m*" --mode average


#for i in {1..3}
#do
#    python3 train_label.py pdtb3 -b 32 --save-path $model_path/v1/m$i --test-set --valid-seed $i \
#                                       --val-metric recall --none-weight 0.01
#done
#
#for i in {1..3}
#do
#    python3 predict_labels.py pdtb3 "$model_path/v${i}/m*" -o $model_path/v${i}/pdtb3.json
#    for j in {1..3}
#    do
#        python3 train_sense.py pdtb3 $model_path/v${i}/pdtb3.json -b 32 --save-path $model_path/v$i/m$j --hidden 256,64
#    done
#    python3 predict_signals.py pdtb3 "$model_path/v${i}/m*" -o $model_path/v${i}/pdtb3.full.json --is-relation-threshold 0.1
#
#    if [[ $i -lt 3 ]]; then
#        python3 predict_signals.py nyt "$model_path/v${i}/m*" -o $model_path/v${i}/nyt.json --limit 5000 --is-relation-threshold 0.1
#        for j in {1..3}
#        do
#            python3 train_label.py pdtb3 --corpus-plus $model_path/v${i}/nyt.json -b 32 \
#                                         --save-path  "$model_path/v$((i+1))/m$j" \
#                                         --valid-seed $i \
#                                         --sort-select \
#                                         --none-weight 0.01\
#                                         --val-metric recall \
#                                         --test-set
#        done
#    fi
#done
#
#
#python3 predict_signals.py nyt "$model_path/v3/m*" -o $model_path/v3/nyt.full.json --limit 10000 --is-relation-threshold 0.1

for p in 0.9 0.8 0.7 0.6 0.5 0.4
do
    for i in {1..5}
    do
        python3 train_label.py pdtb3 --corpus-plus $model_path/v3/nyt.full4k.json -b 32 \
                                     --save-path $model_path/final-s/$p/m${i} \
                                     --document-limit 2500 \
                                     --paragraph-relation-threshold $p \
                                     --valid-seed $i \
                                     --val-metric f1-score \
                                     --test-set
    done
#    python3 test_label.py pdtb3 -b 32 --save-path "$model_path/final/$p/m*" --mode average
done

#python3 test_label.py pdtb3 -b 32 --save-path "$model_path/v1/m*" --mode average
#python3 test_label.py pdtb3 -b 32 --save-path "$model_path/v2/m*" --mode average
#python3 test_label.py pdtb3 -b 32 --save-path "$model_path/v3/m*" --mode average
#python3 test_label.py pdtb3 -b 32 --save-path "$model_path/final/m*" --mode average