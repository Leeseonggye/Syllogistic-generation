from arg_parser import make_parser
from train_utils import train
from utils_gpt2 import set_seed, merge_data, augmented_dataset_merge_kfold

import torch
from torch.utils.data import DataLoader

import os
import pandas as pd
import wandb

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main(args = None):
    if args is None:
        args = make_parser()
    print("current working directory : ", os.getcwd())
    print("gpu device for usage : ", torch.cuda.get_device_name(args.gpu_id))
    
    print("="*20, "Arguments", "="*20)
    print('\n'.join(f'{k:20} : {v}' for k, v in vars(args).items()))
    print("="*45)

    fold_name = f"{args.group_name}-kfold({args.kfold_idx})" if args.num_kfold != 0 else f"{args.group_name}-finetune"
    args.fold_name = fold_name
    wandb.init(
        project = args.project_name,
        name = args.fold_name, 
        reinit = True,
        group = args.group_name)
    
    if args.kfold_idx == 0:
    
        if args.generation_target == 'prem1':
            augmented_dataset_merge_kfold(args,'prem2')
        
        elif args.generation_target == 'prem2':
            augmented_dataset_merge_kfold(args,'prem1')
    

    # # Gradient Accumulation
    # if args.accumulation_steps > 1 :
    #     print("Gradient Accumulation is enabled")
    #     print("Accumulation steps : ", args.accumulation_steps)
    #     print("Batch size : ", args.batch_size)
    #     print("Accumulated Batch size : ", args.batch_size * args.accumulation_steps)
        
    # set_seed(args.seed)      

    # # 저장여부 확인 및 경로 설정
    # if args.save_final_model:
    #     if not os.path.exists(os.path.join(args.generation_path, args.group_name)) :
    #         os.mkdir(os.path.join(args.generation_path, args.group_name))
    #         os.mkdir(os.path.join(args.model_path, args.group_name))
    # else: 
    #     print("="*30)
    #     print("not save at all")
    #     print("="*30)

    # if not os.path.exists(os.path.join(args.log_dir, args.group_name)) :
    #     os.mkdir(os.path.join(args.log_dir, args.group_name))

    # model, tokenizer, Dataset, collate_fn = prepare_model_datasets(args)
    
    # if args.num_augmentation == 0:

    #     # Finetune 시에는 kfold 고려 / pretrain 시에는 고려 X
    #     if args.train_mode == "kfold":
    #         df = pd.read_csv(os.path.join(args.data_path, "Avicenna_train.csv"), encoding = "Windows-1252")
    #         if (args.generation_target == 'conclusion') or (args.generation_target == 'prem2'):
    #             X = df[df["Syllogistic relation"] == "yes"]["Premise 1"].to_list()
    #             y = df[df["Syllogistic relation"] == "yes"]["Syllogistic relation"].to_list()
                
    #         elif args.generation_target == 'prem1':
    #             X = df[df["Syllogistic relation"] == "yes"]["Premise 2"].to_list()
    #             y = df[df["Syllogistic relation"] == "yes"]["Syllogistic relation"].to_list()   

    #         if args.num_kfold != 0 :
    #             from sklearn.model_selection import KFold
    #             splitter = KFold(n_splits = args.num_kfold, shuffle = True, random_state = args.seed)
    #             kfold_split = [(train_idx, val_idx) for train_idx, val_idx in splitter.split(X, y)]
    #             train_idx, val_idx = kfold_split[args.kfold_idx]
    #             train_idx = df[df["Syllogistic relation"] == "yes"].index[train_idx].to_list()
    #             val_idx = df[df["Syllogistic relation"] == "yes"].index[val_idx].to_list()

    #         else:
    #             train_idx = df[df["Syllogistic relation"] == "yes"].index.to_list()
                
    #         train_datasets = Dataset(args, "Avicenna_train.csv", tokenizer, kfold_idx = train_idx)
    #         train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)
        
    #     else : 
    #         train_datasets = Dataset(args, "Avicenna_train.csv", tokenizer)
    #         train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)


    #     args.data_length = len(train_datasets)
    #     wandb.config.update(args)
        

    #     print(f"""
    #     ============================
    #     Train Data Volume : {len(train_datasets)}
    #     ============================
    #     """)
        
    #     if args.num_kfold != 0 :
    #         print("=========================", "Train with Validation Set", "=========================")
    #         val_datasets = Dataset(args, "Avicenna_train.csv", tokenizer, kfold_idx = val_idx)
    #         val_dataloader = DataLoader(val_datasets, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False)
    #         train(args, model, train_dataloader, tokenizer, val_dataloader, val_idx, wandb = wandb)
        
    #     else:
    #         print("=========================", "Train with Test Set", "=========================")
    #         test_dataset = Dataset(args, "Avicenna_test.csv", tokenizer)
    #         test_dataloader = DataLoader(test_dataset, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False) 
    #         train(args, model, train_dataloader, tokenizer, test_dataloader, None, wandb = wandb)
    
    # elif args.num_augmentation > 0:
    #     # if os.path.isfile(os.path.join(args.data_path,f"{args.model_name}_{args.augmentation_method}_num_augmentation_until_{args.num_augmentation}_filtering_{args.filtering}_epoch_{args.prem_making_epochs}.csv")):
    #     #     pass
    #     # else:
    #         # extract_premise(args)
    #     # if args.generation_target != 'conclusion':
    #     #     augmented_dataset_merge_kfold(args,'prem1')
    #     #     augmented_dataset_merge_kfold(args,'prem2')
            
    #     if args.generation_target == 'conclusion':
            
    #         merge_data(args,"Avicenna_train.csv")

    #     if args.kfold_idx == 0:
            
    #         if args.generation_target == 'prem1':
    #             augmented_dataset_merge_kfold(args,'prem2')
            
    #         elif args.generation_target == 'prem2':
    #             augmented_dataset_merge_kfold(args,'prem1')
        
    #     # Finetune 시에는 kfold 고려 / pretrain 시에는 고려 X
    #     if args.train_mode == "kfold":
    #         df = pd.read_csv(os.path.join(args.data_path, "Avicenna_train.csv"), encoding = "Windows-1252")
    #         if (args.generation_target == 'conclusion') or (args.generation_target == 'prem2'):
    #             X = df[df["Syllogistic relation"] == "yes"]["Premise 1"].to_list()
    #             y = df[df["Syllogistic relation"] == "yes"]["Syllogistic relation"].to_list()
                
    #         elif args.generation_target == 'prem1':
    #             X = df[df["Syllogistic relation"] == "yes"]["Premise 2"].to_list()
    #             y = df[df["Syllogistic relation"] == "yes"]["Syllogistic relation"].to_list()
                
    #         if args.num_kfold != 0 :
    #             from sklearn.model_selection import KFold
    #             splitter = KFold(n_splits = args.num_kfold, shuffle = True, random_state = args.seed)
    #             kfold_split = [(train_idx, val_idx) for train_idx, val_idx in splitter.split(X, y)]
    #             train_idx, val_idx = kfold_split[args.kfold_idx]
    #             train_idx = df[df["Syllogistic relation"] == "yes"].index[train_idx].to_list()
    #             val_idx = df[df["Syllogistic relation"] == "yes"].index[val_idx].to_list()
                
    #         else:
    #             train_idx = df[df["Syllogistic relation"] == "yes"].index.to_list()
            
    #         if args.generation_target == 'prem1': # prem1 or prem2
    #             data_name = os.path.join(f"dataset_for_premise_making/{args.model_name}", f"{args.model_name}_prem2_aug_num_{args.num_augmentation}_for_making_prem_epoch_{args.prem_making_epochs}.csv")
    #             train_datasets = Dataset(args, data_name, tokenizer, kfold_idx = train_idx)
    #             train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)
            
    #         elif args.generation_target == 'prem2': # prem1 or prem2
    #             data_name = os.path.join(f"dataset_for_premise_making/{args.model_name}", f"{args.model_name}_prem1_aug_num_{args.num_augmentation}_for_making_prem_epoch_{args.prem_making_epochs}.csv")
    #             train_datasets = Dataset(args, data_name, tokenizer, kfold_idx = train_idx)
    #             train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)
            
    #         else:
    #             train_datasets = Dataset(args, f"{args.model_name}_{args.augmentation_method}_num_augmentation_until_{args.num_augmentation}_filtering_{args.filtering}_epoch_{args.prem_making_epochs}.csv", tokenizer, kfold_idx = train_idx)
    #             train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)
                
        
    #     else : 
    #         train_datasets = Dataset(args, f"{args.model_name}_{args.augmentation_method}_num_augmentation_until_{args.num_augmentation}_filtering_{args.filtering}_epoch_{{args.prem_making_epochs}}.csv", tokenizer)
    #         train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)


    #     args.data_length = len(train_datasets)
    #     wandb.config.update(args)

    #     print(f"""
    #     ============================
    #     Train Data Volume : {len(train_datasets)}
    #     ============================
    #     """)
        
    #     if args.num_kfold != 0 :
    #         if args.generation_target == 'conclusion':
    #             print("=========================", f"Train with Validation Set with Avicenna train", "=========================")
    #             # val_datasets = TestDataset(args, f"{args.model_name}_{args.augmentation_method}_num_augmentation_until_{args.num_augmentation}_filtering_{args.filtering}_epoch_{args.prem_making_epochs}.csv", tokenizer, kfold_idx = val_idx)
    #             # val_dataloader = DataLoader(val_datasets, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False)
                
    #             # validation 정확도를 위해 기존 데이터셋에 있는 데이터만 validation에 사용
    #             val_datasets = Dataset(args, "Avicenna_train.csv", tokenizer, kfold_idx = val_idx)
    #             val_dataloader = DataLoader(val_datasets, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False)
    #             train(args, model, train_dataloader, tokenizer, val_dataloader, val_idx, wandb = wandb)
            
    #         elif args.generation_target == 'prem1':
    #             data_name = os.path.join(f"dataset_for_premise_making/{args.model_name}", f"{args.model_name}_prem2_aug_num_{args.num_augmentation}_for_making_prem_epoch_{args.prem_making_epochs}.csv")
    #             print("=========================", f"Making {args.generation_target} with {args.model_name} num_augmentation_{args.num_augmentation}_epoch_{args.prem_making_epochs}", "=========================")
    #             val_datasets = Dataset(args, data_name, tokenizer, kfold_idx = val_idx)
    #             val_dataloader = DataLoader(val_datasets, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False)
    #             train(args, model, train_dataloader, tokenizer, val_dataloader, val_idx, wandb = wandb)
            
    #         elif args.generation_target == 'prem2':
    #             data_name = os.path.join(f"dataset_for_premise_making/{args.model_name}", f"{args.model_name}_prem1_aug_num_{args.num_augmentation}_for_making_prem_epoch_{args.prem_making_epochs}.csv")
    #             print("=========================", f"Making {args.generation_target} with {args.model_name} num_augmentation_{args.num_augmentation}_epoch_{args.prem_making_epochs}", "=========================")
    #             val_datasets = Dataset(args, data_name, tokenizer, kfold_idx = val_idx)
    #             val_dataloader = DataLoader(val_datasets, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False)
    #             train(args, model, train_dataloader, tokenizer, val_dataloader, val_idx, wandb = wandb)
                
    #     else:
    #         print("=========================", f"Test with train by {args.augmentation_method}_num_augmentation_until_{args.num_augmentation}_filtering_{args.filtering}_epoch_{args.prem_making_epochs}", "=========================")
    #         test_dataset = Dataset(args, "Avicenna_test.csv", tokenizer)
    #         test_dataloader = DataLoader(test_dataset, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False) 
    #         train(args, model, train_dataloader, tokenizer, test_dataloader, None, wandb = wandb)
        
 


    # # Finetune 시에는 kfold 고려 / pretrain 시에는 고려 X
    # if args.train_mode == "kfold":
    #     df = pd.read_csv(os.path.join(args.data_path, "Avicenna_train.csv"), encoding = "Windows-1252")
    #     X = df[df["Syllogistic relation"] == "yes"]["Premise 1"].to_list()
    #     y = df[df["Syllogistic relation"] == "yes"]["Syllogistic relation"].to_list()  

    #     if args.num_kfold != 0 :
    #         from sklearn.model_selection import KFold
    #         splitter = KFold(n_splits = args.num_kfold, shuffle = True, random_state = args.seed)
    #         kfold_split = [(train_idx, val_idx) for train_idx, val_idx in splitter.split(X, y)]
    #         train_idx, val_idx = kfold_split[args.kfold_idx]
    #         train_idx = df[df["Syllogistic relation"] == "yes"].index[train_idx].to_list()
    #         val_idx = df[df["Syllogistic relation"] == "yes"].index[val_idx].to_list()

    #     else:
    #         train_idx = df[df["Syllogistic relation"] == "yes"].index.to_list()
            
    #     train_datasets = Dataset(args, "Avicenna_train.csv", tokenizer, kfold_idx = train_idx)
    #     train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)
    
    # else : 
    #     train_datasets = Dataset(args, "Avicenna_train.csv", tokenizer)
    #     train_dataloader = DataLoader(train_datasets, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True)


    # args.data_length = len(train_datasets)
    # wandb.config.update(args)
    

    # print(f"""
    # ============================
    # Train Data Volume : {len(train_datasets)}
    # ============================
    # """)
    
    # if args.num_kfold != 0 :
    #     print("=========================", "Train with Validation Set", "=========================")
    #     val_datasets = Dataset(args, "Avicenna_train.csv", tokenizer, kfold_idx = val_idx)
    #     val_dataloader = DataLoader(val_datasets, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False)
    #     train(args, model, train_dataloader, tokenizer, val_dataloader, wandb = wandb)
    # else:
    #     print("=========================", "Train with Test Set", "=========================")
    #     test_dataset = Dataset(args, "Avicenna_test.csv", tokenizer)
    #     test_dataloader = DataLoader(test_dataset, batch_size = args.valid_batch_size, collate_fn = collate_fn, shuffle = False) 
    #     train(args, model, train_dataloader, tokenizer, test_dataloader, wandb = wandb)


def prepare_model_datasets(args) :
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    from data_gpt2 import SyllogisticGPT2Dataset, collate_fn
    additional_token_list = ['<|and|>','<|so|>']
    model = GPT2LMHeadModel.from_pretrained(args.load_model_path)
    model.config.max_length = args.max_len
    tokenizer = GPT2Tokenizer.from_pretrained(
        args.load_tokenizer_path, 
        bos_token = '<|startoftext|>', 
        pad_token = '<|pad|>',
        eos_token = '<|endoftext|>',
        additional_special_tokens = additional_token_list)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    
    Dataset = SyllogisticGPT2Dataset
    collate_fn = collate_fn

    return model, tokenizer, SyllogisticGPT2Dataset, collate_fn


if __name__ == "__main__":
    main()