#!/bin/bash

for aug_method in aug_mix aug_pair ori_mix
do
    for num_augmentation in {1..3}
    do  
        for filtering in true
        do
            for prem_making_epoch in {6..9}
            do
                for fold_idx in {0..4}
                do
                    
                python3 ../main.py -group_name KFOLD_VALIDATION_bart_20epochs_${aug_method}_num_augmentation_${num_augmentation}_filtering_${filtering}_prem_making_epoch_${prem_making_epoch} \
                    -num_augmentation $num_augmentation \
                    -bertscore_ceil 0.99 \
                    -bertscore_floor 0.95 \
                    -kfold_idx $fold_idx \
                    -augmentation_method $aug_method \
                    -num_epochs 20 \
                    -filtering $filtering \
                    -prem_making_epochs $prem_making_epoch
                    # -save_epochs False \
                    # -save_final_model False
                    
                done
            done
        done
    done
done


