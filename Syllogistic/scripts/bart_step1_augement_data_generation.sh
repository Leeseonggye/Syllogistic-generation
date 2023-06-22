#!/bin/bash
for prem_making_epoch in {6..9}
do
    for aug_num in {0..3}
    do
        for target in prem1 prem2
        do
            for fold_idx in {0..4}
            do
                
            python3 ../main.py -group_name bart_aug_dataset \
                -generation_target $target \
                -num_augmentation $aug_num \
                -kfold_idx $fold_idx \
                -prem_making_epochs $prem_making_epoch \
                -save_final_model true \

                # -save_epochs False \
                # -save_final_model False
                
            done
        done
    done
done
