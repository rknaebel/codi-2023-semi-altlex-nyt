#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1
export CUDA_LAUNCH_BLOCKING=1
model_path=models/self-ensemble-4

for i in {1..10}
do
    test_random=$[RANDOM]
#    val_random=$[RANDOM]

#    python3 train_label.py pdtb3 -b 32 --save-path $model_path/sota/base/m${i} \
#                                 --valid-seed $val_random \
#                                 --test-seed $test_random \
#                                 --val-metric f1-score \
#                                 --test-set
#    python3 test_label.py pdtb3 -b 32 --save-path "$model_path/sota/base/m${i}" --random-seed $test_random --mode ensemble

#    for j in {1..3}
#    do
#        val_random=$[RANDOM]
#        python3 train_label.py pdtb3 -b 32 --save-path $model_path/sota/ensemble/${i}/m$j \
#                                     --valid-seed $val_random \
#                                     --test-seed $test_random \
#                                     --val-metric f1-score \
#                                     --test-set
#    done
#    python3 test_label.py pdtb3 -b 32 --save-path "$model_path/sota/ensemble/${i}/m*" --random-seed $test_random --mode ensemble

#    python3 train_label.py pdtb3 --corpus-plus $model_path/v3/nyt.full4k.json -b 32 \
#                             --save-path $model_path/sota/final-all/m${i} \
#                             --paragraph-relation-threshold 0.6 \
#                             --valid-seed $val_random \
#                             --test-seed $test_random \
#                             --val-metric f1-score \
#                             --test-set
#    python3 test_label.py pdtb3 -b 32 --save-path "$model_path/sota/final-all/m${i}" --random-seed $test_random > $model_path/sota/final-all/m${i}/test.log

    for j in {1..3}
    do
        val_random=$[RANDOM]
        python3 train_label.py pdtb3 --corpus-plus $model_path/v3/nyt.full4k.json -b 32 \
                                 --save-path $model_path/sota/final-all-ens/$i/m${j} \
                                 --paragraph-relation-threshold 0.6 \
                                 --valid-seed $val_random \
                                 --test-seed $test_random \
                                 --val-metric f1-score \
                                 --test-set
    done
    python3 test_label.py pdtb3 -b 32 --save-path "$model_path/sota/final-all-ens/${i}/m*" --random-seed $test_random --mode ensemble > $model_path/sota/final-all-ens/${i}/test.log

done

