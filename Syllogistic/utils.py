import torch
import numpy as np
import random
import os

import datasets
from numpy import mean

from tqdm import tqdm
import pandas as pd
import gc

import datasets



def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def set_special_token(tokenizer, model, additional_special_tokens):
    special_token_dict = {'additional_special_tokens' : additional_special_tokens}
    tokenizer.add_special_tokens(special_token_dict)
    model.resize_token_embeddings(len(tokenizer))
    print("-"*15, "Special Token 추가 완료", "-"*15)

def train_iter(wandb, args, model, model_inputs, optimizer, step, num_iter, epoch, device, scheduler = None) :
    model_inputs = {key : value.to(device) for key, value in model_inputs.items()}
    outputs = model(**model_inputs)

    loss = outputs.loss/args.accumulation_steps
    loss.backward()

    if (num_iter+1) % args.accumulation_steps == 0 :
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        loss = loss.detach().cpu().item()
        if scheduler:
            scheduler.step()

        wandb.log({"train_loss" : loss, "epoch" : epoch, "steps" : step})
    return step

def log_validation(args, model, val_dataloader, tokenizer, index,wandb, epoch = 0) :
    if args.valid_epochs:            
        (val_rouge1, val_rouge2, val_rougeL), val_bleu, val_generation, val_loss = test(args, model, val_dataloader, tokenizer, index)

        log = {
            'val_bleu' : val_bleu, 
            'val_rouge1' : val_rouge1, 
            'val_rouge2' : val_rouge2,
            'val_rougeL' : val_rougeL,
            'val_loss' : val_loss
            }
        wandb.log(log)
        save_log(args, log, epoch)

    if args.generation_target == 'conclusion':
        
        if args.num_kfold ==0:
            kfold_result(args)
            best_epoch = extract_best_epoch(args)
            print(f"kfold_num은 0 best epoch: {best_epoch}")
            
        else:
            best_epoch = args.num_epochs
            print(f"Kfold_num은 5 best epoch: {best_epoch}")
        
        if (args.save_final_model)& (epoch == best_epoch - 1):
            print("-"*10, "final model & tokenizer & generation result saved", "-"*10)
            save_generation(val_generation, args, epoch = epoch)
            save_model(model, tokenizer, args, epoch = epoch)
        
    elif (args.generation_target != 'conclusion')&(args.save_final_model)& (epoch == args.prem_making_epochs - 1):
        print("-"*10, "final model & tokenizer & generation result saved", "-"*10)
        save_generation(val_generation, args, epoch = epoch)
        save_model(model, tokenizer, args, epoch = epoch)
    
    
    

    print(f""" 
    ----------------Validation Result----------------
    --------------------------------------------------
    | Epoch | BLEU | ROUGE1 | ROUGE2 | ROUGEL | Loss |
    --------------------------------------------------
    | {round(epoch+1,4)} | {round(val_bleu,4)} | {round(val_rouge1,4)} | {round(val_rouge2,4)} | {round(val_rougeL,4)} | {round(val_loss,4)} |
    --------------------------------------------------
    """)


def test(args, model, dataloader, tokenizer,index) :
    model.eval()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    index_list = index
    generation_list = []
    label_list = []
    input_list = []
    loss_list = []

    scorer = Score_Calculator(tokenizer)

    with torch.no_grad():
        print("-"*15, f"Test", "-"*15)
        for model_inputs in tqdm(dataloader) :
            model_inputs = {key : value.to(device) for key, value in model_inputs.items()}
            # calculate loss
            loss = model(**model_inputs).loss.detach().cpu().item()
            model.zero_grad()
            loss_list.append(loss)

            # calculate bleu, rouge score
            generation = model.generate(
                input_ids = model_inputs["input_ids"], 
                attention_mask = model_inputs["attention_mask"],
                max_length = args.max_len
                ).cpu()  

            labels = model_inputs["labels"].cpu()
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)

            # save generation & label string
            inputs, labels, generation = tensor2text(tokenizer, model_inputs["input_ids"].cpu(), labels, generation)
            scorer.update_corpus(labels, generation)
            input_list.extend(inputs)
            generation_list.extend(generation)
            label_list.extend(labels)
    

    model.train()
    
    if args.generation_target == 'conclusion':
        
        generation_df = pd.DataFrame({
            "input" : input_list,
            "generation" : generation_list,
            "label" : label_list})
    
    else: 
        generation_df = pd.DataFrame({
            "index" : index_list,
            "input" : input_list,
            "generation" : generation_list,
            "label" : label_list})
    
    valid_loss = sum(loss_list)/len(dataloader)

    (rouge1, rouge2, rougeL), (bleu) = scorer.get_score()
    
    del scorer, model_inputs, loss_list, generation_list, label_list, input_list
    gc.collect()
    torch.cuda.empty_cache()
    
    return (rouge1, rouge2, rougeL), bleu, generation_df, valid_loss


class Score_Calculator():
    def __init__(self, tokenizer) :
        """
        calculator = Score_Calculator(tokenizer)
        for batch in test_dataloader:
            generation = model.generate(**batch)
            labels = batch["labels"]

            calculator.update_score(labels, generation)
        bleu, rouge = calculator.get_final_score()
        """
        self.rouge_scorer = datasets.load_metric("rouge")
        self.rouge_score = 0

        self.bleu_scorer = datasets.load_metric("bleu")
        self.bleu_score = 0

        self.tokenizer = tokenizer

        self.label_corpus = []
        self.prediction_corpus = []

    def update_corpus(self, label : list, prediction : list):
        """
        label, prediction : list of string
        """
        self.label_corpus.extend(label)
        self.prediction_corpus.extend(prediction)

    def get_score(self) :
        print(f"label : {len(self.label_corpus)}, prediction : {len(self.prediction_corpus)}")
        self.rouge_score = self.rouge_scorer.compute(
            predictions = self.prediction_corpus,
            references = self.label_corpus,
        )

        bleu_prediction = [pred.split() for pred in self.prediction_corpus]
        bleu_label = [[lab.split()] for lab in self.label_corpus]
        self.bleu_score = self.bleu_scorer.compute(
            predictions = bleu_prediction,
            references = bleu_label,
        )

        return (self.rouge_score["rouge1"].mid.fmeasure, self.rouge_score["rouge2"].mid.fmeasure, self.rouge_score["rougeL"].mid.fmeasure), self.bleu_score["bleu"]

def save_generation(file, args, epoch = None) :
    if args.generation_target == 'conclusion':
        file.to_html(os.path.join(args.generation_path, args.group_name, f"{args.group_name}_epoch_{epoch}.html"), index = False)
    
    else: # premise 생성하는 경우 
        file.to_html(os.path.join(args.generation_path, f'aug_dataset/{args.model_name}', f"{args.generation_target}_aug_num_{args.num_augmentation + 1}_{args.model_name}_kfold_{args.kfold_idx}_epoch_{epoch+1}.html"), index = False)
        file.to_csv(os.path.join(args.generation_path, f'aug_dataset/{args.model_name}', f"{args.generation_target}_aug_num_{args.num_augmentation + 1}_{args.model_name}_kfold_{args.kfold_idx}_epoch_{epoch+1}.csv"), index = False)
        
# 모델 저장 코드 수정 
def save_model(model, tokenizer, args, epoch = None):
    save_path = os.path.join(args.model_path, args.group_name)
    model.save_pretrained(os.path.join(save_path, f"{args.model_save_name}_{args.group_name}"))
    tokenizer.save_pretrained(save_path)

def load_log_df(args):
    if not os.path.exists(os.path.join(args.log_dir, args.group_name, f"log-{args.group_name}_kfold({args.kfold_idx}).csv")):
        log_df = pd.DataFrame(columns = ["epoch", "val_bleu", "val_rouge", "val_loss"])
        log_df.to_csv(os.path.join(args.log_dir, args.group_name, f"log-{args.group_name}_kfold({args.kfold_idx}).csv"), index = False)
    else :
        log_df = pd.read_csv(os.path.join(args.log_dir, args.group_name, f"log-{args.group_name}_kfold({args.kfold_idx}).csv"))
    return log_df

def save_log(args, log, epoch = None):
    log["epoch"] = epoch
    log["data_length"] = args.data_length
    if args.num_kfold != 0 :
        log["kfold"] = args.kfold_idx
    log_df_new = pd.DataFrame(log, index = [0])
    log_df = load_log_df(args)
    log_df = log_df.append(log_df_new, ignore_index = True)
    log_df.to_csv(os.path.join(args.log_dir, args.group_name, f"log-{args.group_name}_kfold({args.kfold_idx}).csv"), index = False)

def tensor2text(tokenizer, inputs, label, generation):
    inputs = tokenizer.batch_decode(inputs, skip_special_tokens = True)
    label = tokenizer.batch_decode(label, skip_special_tokens = True)
    generation = tokenizer.batch_decode(generation, skip_special_tokens = True)
    return inputs, label, generation



def augmented_dataset_merge_kfold(args,target):
    path = os.path.join(args.generation_path, f'aug_dataset/{args.model_name}')
    df = pd.read_csv(os.path.join(args.data_path, "Avicenna_train.csv"), encoding = 'Windows-1252')
    df_yes = df[df["Syllogistic relation"]=='yes']
    df_yes['index'] = df_yes.index
    conc_list = df_yes['Conclusion'].values
    model = args.model_name
    bert_scorer = datasets.load_metric('bertscore')
    for fold in range(5):
        df = pd.read_csv(os.path.join(path,f"{target}_aug_num_{args.num_augmentation}_{model}_kfold_{fold}_epoch_{args.prem_making_epochs}.csv"))
        
        if fold == 0:
            total = df
        
        else:
            total = pd.concat([total, df], axis = 0)
            

        total = total.sort_values('index')
        
    total['bert_f1'] = 0
    

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
    
    total.to_csv(os.path.join(args.data_path, f"augmented_dataset_raw/{args.model_name}/{model}_{target}_aug_num_{args.num_augmentation}_epoch_{args.prem_making_epochs}.csv"))
    total.to_html(os.path.join(args.data_path, f"augmented_dataset_raw/{args.model_name}/{model}_{target}_aug_num_{args.num_augmentation}_epoch_{args.prem_making_epochs}.html"))
    
    aug_raw = pd.read_csv(os.path.join(args.data_path, f"augmented_dataset_raw/{args.model_name}/{args.model_name}_{target}_aug_num_{args.num_augmentation}_epoch_{args.prem_making_epochs}.csv"))
    
    if target == 'prem1':
        concat_df = pd.merge(aug_raw[['index','Premise 1', 'Premise 1_f1']],df_yes[['index','Premise 2','Syllogistic relation','Conclusion']], how = 'inner', on = 'index')

    elif target == 'prem2':
        concat_df = pd.merge(aug_raw[['index','Premise 2', 'Premise 2_f1']],df_yes[['index','Premise 1','Syllogistic relation','Conclusion']], how = 'inner', on = 'index')

    concat_df.to_csv(os.path.join(args.data_path,f"dataset_for_premise_making/{args.model_name}",f"{args.model_name}_{target}_aug_num_{args.num_augmentation}_for_making_prem_epoch_{args.prem_making_epochs}.csv"))
        
    return total

    
        

def merge_data(args, original_data_name) :

    data_path = os.path.join(args.data_path, f"augmented_dataset_raw/{args.model_name}")
    save_path = args.data_path
    augmentation_method = args.augmentation_method
    num_augmentation = args.num_augmentation
    bertscore_ceil = args.bertscore_ceil
    bertscore_floor = args.bertscore_floor

    original_data_path = os.path.join(args.data_path, original_data_name)
    original_data = pd.read_csv(original_data_path, encoding = "Windows-1252")
    original_data["index"] = original_data.index
    original_data = original_data[original_data['Syllogistic relation'] == 'yes']

    if augmentation_method == 'aug_pair':

        for i in range(1, num_augmentation+1):
        
            prem1_data_path = os.path.join(data_path, f"{args.model_name}_prem1_aug_num_{i}_epoch_{args.prem_making_epochs}.csv")
            prem1_data = pd.read_csv(prem1_data_path)
            prem2_data_path = os.path.join(data_path, f"{args.model_name}_prem2_aug_num_{i}_epoch_{args.prem_making_epochs}.csv")
            prem2_data = pd.read_csv(prem2_data_path)

            augmented_data = pd.merge(prem1_data[["index","Premise 1","Premise 1_f1"]],prem2_data[["index","Premise 2","Premise 2_f1","Conclusion"]], how = 'inner', on = 'index')
            augmented_data["Syllogistic relation"] = "yes"

            if args.filtering == 'false':
                augmented_data = augmented_data[["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]
            
            elif args.filtering == 'true':

                condition = (augmented_data["Premise 1_f1"] < bertscore_ceil) & (augmented_data["Premise 1_f1"] > bertscore_floor) & (augmented_data["Premise 2_f1"] < bertscore_ceil) & (augmented_data["Premise 2_f1"] > bertscore_floor)
                augmented_data = augmented_data[condition][["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]

            else:
                print("잘못된 입력 입니다.")

            if i == 1:
                total_data = pd.concat([original_data, augmented_data], axis=0)
            
            else:
                total_data = pd.concat([total_data, augmented_data],axis=0)
            
            total_data.to_csv(os.path.join(save_path, f"{args.model_name}_{augmentation_method}_num_augmentation_until_{i}_filtering_{args.filtering}_epoch_{args.prem_making_epochs}.csv"), index = False)

    elif augmentation_method == 'ori_mix':
        
        for i in range(1, num_augmentation+1):
        
            prem1_data_path = os.path.join(data_path, f"{args.model_name}_prem1_aug_num_{i}_epoch_{args.prem_making_epochs}.csv")
            prem1_data = pd.read_csv(prem1_data_path)
            prem2_data_path = os.path.join(data_path, f"{args.model_name}_prem2_aug_num_{i}_epoch_{args.prem_making_epochs}.csv")
            prem2_data = pd.read_csv(prem2_data_path)

            prem1_aug = pd.merge(prem1_data[["index","Premise 1","Premise 1_f1"]], original_data[["index","Premise 2", "Conclusion"]], how = 'inner', on = 'index')
            prem1_aug["Syllogistic relation"] = "yes"
            prem2_aug = pd.merge(original_data[["index","Premise 1"]], prem2_data[["index","Premise 2","Premise 2_f1","Conclusion"]], how = 'inner', on = 'index')
            prem2_aug["Syllogistic relation"] = "yes"

            if args.filtering == 'false':
                prem1_aug = prem1_aug[["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]
                prem2_aug = prem2_aug[["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]
                augmented_data = pd.concat([prem1_aug,prem2_aug])

            elif args.filtering == 'true':
                condition_p1 = (prem1_aug["Premise 1_f1"] < bertscore_ceil) & (prem1_aug["Premise 1_f1"] > bertscore_floor)
                condition_p2 = (prem2_aug["Premise 2_f1"] < bertscore_ceil) & (prem2_aug["Premise 2_f1"] > bertscore_floor)
                prem1_aug = prem1_aug[condition_p1][["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]
                prem2_aug = prem2_aug[condition_p2][["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]
                augmented_data = pd.concat([prem1_aug,prem2_aug])

            else:
                print("잘못된 입력 입니다.")


            if i == 1:
                total_data = pd.concat([original_data, augmented_data])
            
            else:
                total_data = pd.concat([total_data, augmented_data])

            total_data.to_csv(os.path.join(save_path, f"{args.model_name}_{augmentation_method}_num_augmentation_until_{i}_filtering_{args.filtering}_epoch_{args.prem_making_epochs}.csv"), index = False)



    elif augmentation_method == 'aug_mix':
        for i in range(1, num_augmentation+1):
        
            prem1_data_path = os.path.join(data_path, f"{args.model_name}_prem1_aug_num_{i}_epoch_{args.prem_making_epochs}.csv")
            prem1_data = pd.read_csv(prem1_data_path)
            prem2_data_path = os.path.join(data_path, f"{args.model_name}_prem2_aug_num_{i}_epoch_{args.prem_making_epochs}.csv")
            prem2_data = pd.read_csv(prem2_data_path)

            prem1_ori = pd.merge(prem1_data[["index","Premise 1","Premise 1_f1"]], original_data[["index","Premise 2","Conclusion"]], how = 'inner', on = 'index')
            prem1_ori["Syllogistic relation"] = "yes"
            prem2_ori = pd.merge(original_data[["index","Premise 1"]], prem2_data[["index","Premise 2","Premise 2_f1","Conclusion"]], how = 'inner', on = 'index')
            prem2_ori["Syllogistic relation"] = "yes"
            aug_pair = pd.merge(prem1_data[["index","Premise 1","Premise 1_f1"]],prem2_data[["index","Premise 2","Premise 2_f1","Conclusion"]], how = 'inner', on = 'index')
            aug_pair["Syllogistic relation"] = "yes"
            
            if args.filtering == 'false':
                prem1_ori = prem1_ori[["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]
                prem2_ori = prem2_ori[["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]
                aug_pair = aug_pair[["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]
                augmented_data = pd.concat([prem1_ori,prem2_ori,aug_pair])
            
            elif args.filtering == 'true':
                condition_p1 = (prem1_ori["Premise 1_f1"] < bertscore_ceil) & (prem1_ori["Premise 1_f1"] > bertscore_floor)
                condition_p2 = (prem2_ori["Premise 2_f1"] < bertscore_ceil) & (prem2_ori["Premise 2_f1"] > bertscore_floor)
                condition_pair = (aug_pair["Premise 1_f1"] < bertscore_ceil) & (aug_pair["Premise 1_f1"] > bertscore_floor) & (aug_pair["Premise 2_f1"] < bertscore_ceil) & (aug_pair["Premise 2_f1"] > bertscore_floor)
                prem1_ori = prem1_ori[condition_p1][["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]
                prem2_ori = prem2_ori[condition_p2][["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]
                aug_pair = aug_pair[condition_pair][["Premise 1", "Premise 2", "Conclusion", "Syllogistic relation", "index"]]
                augmented_data = pd.concat([prem1_ori,prem2_ori,aug_pair])
            
            else:
                print("잘못된 입력입니다.")

            if i == 1:
                total_data = pd.concat([original_data, augmented_data])
            
            else:
                total_data = pd.concat([total_data, augmented_data])
            
            total_data.to_csv(os.path.join(save_path, f"{args.model_name}_{augmentation_method}_num_augmentation_until_{i}_filtering_{args.filtering}_epoch_{args.prem_making_epochs}.csv"), index = False)

    return total_data

# Kfold result 정리

def kfold_result(args):
    for i in range(5):
        if args.model_name != 'bart':
            k_fold_path = os.path.join(args.log_dir,f"KFOLD_VALIDATION_20epochs_{args.model_name}_{args.augmentation_method}_num_augmentation_{args.num_augmentation}_filtering_{args.filtering}_prem_making_epoch_{args.prem_making_epochs}/log-KFOLD_VALIDATION_20epochs_{args.model_name}_{args.augmentation_method}_num_augmentation_{args.num_augmentation}_filtering_{args.filtering}_prem_making_epoch_{args.prem_making_epochs}_kfold({i}).csv" )
        
        else: 
            k_fold_path = os.path.join(args.log_dir,f"KFOLD_VALIDATION_{args.model_name}_20epochs_{args.augmentation_method}_num_augmentation_{args.num_augmentation}_filtering_{args.filtering}_prem_making_epoch_{args.prem_making_epochs}/log-KFOLD_VALIDATION_{args.model_name}_20epochs_{args.augmentation_method}_num_augmentation_{args.num_augmentation}_filtering_{args.filtering}_prem_making_epoch_{args.prem_making_epochs}_kfold({i}).csv" )
        log = pd.read_csv(k_fold_path)
        log_columns = log[["epoch",'val_bleu', 'val_loss', 'val_rouge1','val_rouge2', 'val_rougeL']]
        log_filtered = log_columns[-21:]
        
        if i == 0:
            total_df = log_filtered
        else:
            total_df = total_df + log_filtered
        
    total_df = total_df/5 # fold 수
    total_df["bleu_rouge1,2,l"] = (total_df["val_bleu"]+total_df["val_rouge1"]+total_df["val_rouge2"]+total_df["val_rougeL"])
    total_df["bleu_rougel"] = (total_df["val_bleu"]+total_df["val_rougeL"])
    # total_df = total_df[:21]
    total_df.to_csv(os.path.join(args.log_dir, args.group_name, f"log-{args.group_name}_kfold_blue_rouge_combination_final.csv"))

    return total_df

def extract_best_epoch(args):
    result = pd.read_csv(os.path.join(args.log_dir, args.group_name, f"log-{args.group_name}_kfold_blue_rouge_combination_final.csv"))
    if args.best_epoch_method == 'rougeL':
        best_epoch = result['bleu_rougel'].argmax()
    elif args.best_epoch_method == 'rouge12L':
        best_epoch = result['bleu_rouge1,2,l'].argmax()
    else:
        print("Wrong input")
            
    return best_epoch