#!/bin/bash

for aug_method in ori_mix aug_mix aug_pair
do
    for num_augmentation in {1..3}
    do
        for filtering in true 
        do
            for prem_making_epoch in {8..9}
            do         
            python3 calculate_scores.py -num_augmentation $num_augmentation \
                -language_model 'gpt2' \
                -method $aug_method \
                -filtering $filtering \
                -prem_making_epoch $prem_making_epoch
            done
        done
    done
done

