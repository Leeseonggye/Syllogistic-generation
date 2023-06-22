#!/bin/bash
for prem_making_epoch in {6..9}
do
    for aug_num in {1..3}
    do
        for target in prem1 prem2
        do
            for fold_idx in {0..4}
            do
                
            python3 ../main.py -group_name t5_aug_dataset \
                -learning_rate 3e-4 \
                -generation_target $target \
                -num_augmentation $aug_num \
                -kfold_idx $fold_idx \
                -prem_making_epochs $prem_making_epoch \
                -save_final_model true \
                -model_name 't5' \
                -load_model_path 't5-small' \
                -load_tokenizer_path 't5-small' 

                # -save_epochs False \
                # -save_final_model False
                
            done
        done
    done
done
