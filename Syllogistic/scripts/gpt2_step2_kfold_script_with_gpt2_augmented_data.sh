#!/bin/bash

for aug_method in ori_mix aug_mix aug_pair
do
    for num_augmentation in {1..3}
    do  
        for filtering in true
        do
            for prem_making_epoch in {8..9}
            do
                for fold_idx in {0..4}
                do
                    
                python3 ../gpt2_main.py -group_name KFOLD_VALIDATION_20epochs_gpt2_${aug_method}_num_augmentation_${num_augmentation}_filtering_${filtering}_prem_making_epoch_${prem_making_epoch} \
                    -prem_making_epochs $prem_making_epoch \
                    -num_augmentation $num_augmentation \
                    -bertscore_ceil 0.99 \
                    -bertscore_floor 0.95 \
                    -kfold_idx $fold_idx \
                    -augmentation_method $aug_method \
                    -num_epochs 20 \
                    -max_len 512 \
                    -batch_size 4 \
                    -accumulation_steps 4 \
                    -filtering $filtering \
                    -valid_batch_size 8 \
                    -model_name 'gpt2' \
                    -load_model_path 'gpt2' \
                    -load_tokenizer_path 'gpt2' \
                    -gpu_id 1
                    # -save_epochs False \
                    # -save_final_model False
                done
            done
        done
    done
done


