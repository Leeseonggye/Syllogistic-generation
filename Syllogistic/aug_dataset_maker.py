import torch
import numpy as np
import random
import os

import datasets
from numpy import mean

from tqdm import tqdm
import pandas as pd
import gc
from arg_parser import make_parser

import datasets


def augmented_dataset_merge_kfold(args):
    path = os.path.join("syllogistic/syllogistic-generation/generation_log/", f"t5_aug_dataset")
    df = pd.read_csv(os.path.join("syllogistic/syllogistic-generation/datasets/", "Avicenna_train.csv"), encoding = 'Windows-1252')
    df_yes = df[df["Syllogistic relation"] == 'yes']
    conc_list = df_yes['Conclusion'].values
    target = 'prem1'
    model = 't5'
    for fold in range(5):
        df = pd.read_csv(os.path.join(path,f"{target}_aug_num_{args.num_augmentation+1}_{model}_kfold_{fold}_epoch_14.csv"))
        
        if fold == 0:
            total = df
        
        else:
            total = pd.concat([total, df], axis = 0)
            

        total = total.sort_values('index')
        
    total['bert_f1'] = 0
    bert_scorer = datasets.load_metric('bertscore')

    for i in range(len(total)):
        generation_list = []
        label_list = []
        generation_list.append(total['generation'].iloc[i])
        label_list.append(total['label'].iloc[i])
        score = bert_scorer.compute(
            references = label_list, 
            predictions = generation_list, 
            lang = 'en', 
            verbose = True
            )
        
        total['bert_f1'].iloc[i] = score['f1'][0]
        
    if target == 'prem1':
        total = total[['index', 'generation', 'bert_f1']]
        total.rename(columns={'generation':'Premise 1','bert_f1':'Premise 1_f1'},inplace=True)
        
    elif target == 'prem2':
        total = total[['index', 'generation', 'bert_f1']]
        total['Conclusion'] = conc_list
        total.rename(columns={'generation':'Premise 2','bert_f1':'Premise 2_f1'},inplace=True)    
        
    total.to_csv(os.path.join("syllogistic/syllogistic-generation/datasets/", f"augmented_dataset_raw/{model}_{target}_aug_num_{args.num_augmentation}.csv"))
    total.to_html(os.path.join("syllogistic/syllogistic-generation/datasets/", f"augmented_dataset_raw/{model}_{target}_aug_num_{args.num_augmentation}.html"))
            
    return total
    
args = make_parser()
augmented_dataset_merge_kfold(args)