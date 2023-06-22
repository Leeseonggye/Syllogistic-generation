#!/bin/bash

for aug_method in ori_mix aug_mix aug_pair
do
    for num_augmentation in {1..3}
    do
        for filtering in true
        do
            for prem_making_epoch in {6..9}
            do
            python3 ../main.py -group_name AA_final_test_20epochs_t5_rouge12L_${aug_method}_num_augmentation_${num_augmentation}_prem_making_epoch_${prem_making_epoch}_filtered_${filtering} \
                -learning_rate 3e-4 \
                -save_final_model true \
                -num_kfold 0 \
                -num_augmentation $num_augmentation \
                -bertscore_ceil 0.99 \
                -bertscore_floor 0.95 \
                -filtering $filtering \
                -augmentation_method $aug_method \
                -prem_making_epochs $prem_making_epoch \
                -gpu_id 0 \
                -model_name 't5' \
                -load_model_path 't5-small' \
                -load_tokenizer_path 't5-small' 
                # -save_epochs False \
                # -save_final_model False
            done
        done    
    done
done

