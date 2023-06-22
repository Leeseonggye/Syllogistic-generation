import torch
from torch.optim import AdamW

import gc
import os
import wandb
from tqdm import tqdm
import pandas as pd
from arg_parser import make_parser

args = make_parser()

if args.model_name == 'gpt2':
    from utils_gpt2 import Score_Calculator, save_model, log_validation, test, train_iter, kfold_result, extract_best_epoch

else:
    from utils import Score_Calculator, save_model, log_validation, test, train_iter, kfold_result, extract_best_epoch


def train(args, model, dataloader, tokenizer,val_dataloader, index ,scheduler = None, wandb = None) :
    print("="*20, f"Fold : {args.group_name}", "="*20)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    optimizer = AdamW(model.parameters(), lr = args.learning_rate)
    model.train()
    model.to(device)

    if args.num_kfold == 0:
        kfold_result(args)
        epochs = extract_best_epoch(args)
         
    else:
        if args.generation_target == 'conclusion':
            epochs = args.num_epochs
        
        else:
            epochs = args.prem_making_epochs

    step = 0
    log_validation(args, model, val_dataloader, tokenizer, index,wandb, epoch = -1) # 학습 전 validation loss 기록

    for epoch in range(epochs) :
        print("-"*15, f"Epoch : {epoch+1}/ {epochs}", "-"*15)
        print("-"*15, f"Train", "-"*15)
        for num_iter, model_inputs in enumerate(tqdm(dataloader)) :
            step = train_iter(wandb, args, model, model_inputs, optimizer, step, num_iter, epoch, device)

            if step == args.num_iteration : # 목표 step 도달 시 학습 종료 
                print(f"""
                --------------------------------------------------
                목표 iteration 도달, 학습 종료
                --------------------------------------------------
                iteration : {step+1}
                epoch : {epoch}
                """)
                
                break
            
        if args.valid_epochs :
            log_validation(args, model, val_dataloader, tokenizer, index,wandb,epoch) # 매 에폭마다 validation loss 기록

        if step == args.num_iteration :
            log_validation(args, model, val_dataloader, tokenizer, index,wandb,epoch) # 매 에폭마다 validation loss 기록
            break
