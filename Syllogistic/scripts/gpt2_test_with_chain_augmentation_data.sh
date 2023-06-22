#!/bin/bash

for aug_method in ori_mix aug_mix aug_pair
do
    for num_augmentation in {1..3}
    do
        for filtering in true
        do
            for prem_making_epoch in {6..9}
            do
            python3 ../gpt2_main.py -group_name AA_final_test_20epochs_gpt2_rouge12L_${aug_method}_num_augmentation_${num_augmentation}_prem_making_epoch_${prem_making_epoch}_filtered_${filtering} \
                -save_final_model true \
                -num_kfold 0 \
                -num_augmentation $num_augmentation \
                -bertscore_ceil 0.99 \
                -bertscore_floor 0.95 \
                -filtering $filtering \
                -augmentation_method $aug_method \
                -prem_making_epochs $prem_making_epoch \
                -max_len 512 \
                -batch_size 4 \
                -accumulation_steps 4 \
                -valid_batch_size 8 \
                -gpu_id 1 \
                -model_name 'gpt2' \
                -load_model_path 'gpt2' \
                -load_tokenizer_path 'gpt2' \
                # -save_epochs False \
                # -save_final_model False
            done
        done    
    done
done

