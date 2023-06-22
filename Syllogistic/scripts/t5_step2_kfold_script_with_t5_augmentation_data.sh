#!/bin/bash

for aug_method in ori_mix
do
    for num_augmentation in 1 2 3
    do  
        for filtering in true
        do
            for prem_making_epoch in 8
            do
                for fold_idx in 0 1 2 3 4
                do
                    
                python3 ../main.py -group_name KFOLD_VALIDATION_20epochs_t5_${aug_method}_num_augmentation_${num_augmentation}_filtering_${filtering}_prem_making_epoch_${prem_making_epoch} \
                    -learning_rate 3e-4 \
                    -prem_making_epochs $prem_making_epoch \
                    -num_augmentation $num_augmentation \
                    -bertscore_ceil 0.99 \
                    -bertscore_floor 0.95 \
                    -kfold_idx $fold_idx \
                    -augmentation_method $aug_method \
                    -num_epochs 20 \
                    -filtering $filtering \
                    -model_name 't5' \
                    -load_model_path 't5-small' \
                    -load_tokenizer_path 't5-small' \
                    -gpu_id 1
                    # -save_epochs False \
                    # -save_final_model False
                done
            done
        done
    done
done


