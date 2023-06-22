import torch
import os
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class SyllogisticGPT2Dataset(Dataset):  
    def __init__(self, args, data, tokenizer, kfold_idx = None):
        self.data_path = os.path.join(args.data_path, data)
        # self.raw_data = pd.read_csv(self.data_path, encoding = 'Windows-1252')
        self.data_path = os.path.join(args.data_path, data)

        
        if 'Avicenna' in data:
            self.raw_data = pd.read_csv(self.data_path, encoding = 'Windows-1252')
            self.raw_data["index"] = self.raw_data.index
            
        else:
            self.raw_data = pd.read_csv(self.data_path)
            # self.raw_data = self.raw_data.drop_duplicates()
            # print("duplication 제거")
            
        self.raw_data = self.raw_data[self.raw_data["Syllogistic relation"] == "yes"]

        self.max_len = args.max_len
        self.batch_size = args.batch_size
        
        self.tokenizer = tokenizer
        self.index = self.raw_data["index"]

        self.bos = tokenizer.bos_token
        self.eos = tokenizer.eos_token
        self.prem_connect = tokenizer.additional_special_tokens[0] # '<|and|>'
        self.conc = tokenizer.additional_special_tokens[1] #'<|so|>'
        self.input_pad_id = tokenizer.pad_token_id
        self.label_pad_id = -100
        
        self.prem1 = self.raw_data["Premise 1"].to_list()
        self.prem2 = self.raw_data["Premise 2"].to_list()
        self.label = self.raw_data["Conclusion"].to_list()
        
        if kfold_idx is not None:
            self.prem1 = [sent for num, sent in zip(self.index, self.prem1) if num in kfold_idx]
            self.prem2 = [sent for num, sent in zip(self.index, self.prem2) if num in kfold_idx]
            self.label = [sent for num, sent in zip(self.index, self.label) if num in kfold_idx]
            
            print(f">>> generation_target : {args.generation_target} | kfold_idx : {kfold_idx[:10]}")

        assert len(self.prem1) == len(self.prem2) and len(self.prem2) == len(self.label),f"데이터 길이가 다름 \n Premise 1 : {len(self.prem1)} \n Premise 2 : {len(self.prem2)} \n Label : {len(self.label)}"

        if args.generation_target == 'conclusion':
            self.input_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + p2 +  f" {self.conc}"+ label + f"{self.eos}"  for p1, p2, label in zip(self.prem1, self.prem2, self.label)]
            self.label_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + p2 +  f" {self.conc}"+ label + f"{self.eos}"  for p1, p2, label in zip(self.prem1, self.prem2, self.label)]
            self.input_for_generation_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + p2 +  f" {self.conc}"  for p1, p2 in zip(self.prem1, self.prem2)]
            self.conclusion_text = [label for label in self.label]
        
        elif args.generation_target == 'prem1':
            self.input_text = [f"{self.bos} " +p2 + f"{self.prem_connect} " + label +  f" {self.conc}"+ p1 + f"{self.eos}"  for p1, p2, label in zip(self.prem1, self.prem2, self.label)]
            self.label_text = [f"{self.bos} " +p2 + f"{self.prem_connect} " + label +  f" {self.conc}"+ p1 + f"{self.eos}"  for p1, p2, label in zip(self.prem1, self.prem2, self.label)]
            self.input_for_generation_text = [f"{self.bos} " +p2 + f"{self.prem_connect} " + label +  f" {self.conc}"  for p2, label in zip(self.prem2, self.label)]
            self.conclusion_text = [p1 for p1 in self.prem1]
        
        elif args.generation_target == 'prem2':
            self.input_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + label +  f" {self.conc}"+ p2 + f"{self.eos}"  for p1, p2, label in zip(self.prem1, self.prem2, self.label)]
            self.label_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + label +  f" {self.conc}"+ p2 + f"{self.eos}"  for p1, p2, label in zip(self.prem1, self.prem2, self.label)]
            self.input_for_generation_text = [f"{self.bos} " +p1 + f"{self.prem_connect} " + label +  f" {self.conc}"  for p1, label in zip(self.prem1, self.label)]
            self.conclusion_text = [p2 for p2 in self.prem2]
        
    def __len__(self):
        return len(self.input_text)
        
    
    def __getitem__(self, idx):
        return self.__preprocess(self.input_text[idx], self.label_text[idx], self.input_for_generation_text[idx], self.conclusion_text[idx])
    
    def __preprocess(self, input_text, label_text, input_for_generation_text, conclusion_text):
        input_token_ids = torch.full((1, self.max_len), fill_value = self.input_pad_id)
        label_token_ids = torch.full((1, self.max_len), fill_value = self.label_pad_id)
        conclusion_token_ids = torch.full((1, self.max_len), fill_value = self.label_pad_id)
        

        attn_mask = torch.zeros((1, self.max_len))

        self.tokenizer.padding_side = "right"
        input_tokens = self.tokenizer.encode(input_text, add_special_tokens = True, return_tensors = 'pt')
        label_tokens = self.tokenizer.encode(label_text, add_special_tokens = True, return_tensors = 'pt')
        conclusion_tokens = self.tokenizer.encode(conclusion_text, add_special_tokens = True, return_tensors = 'pt')
        
        input_token_ids[0, :input_tokens.shape[1]] = input_tokens
        label_token_ids[0, :label_tokens.shape[1]] = label_tokens

        self.tokenizer.padding_side = "left"
        # input_for_generation = self.tokenizer(input_for_generation_text, add_special_tokens = True, return_tensors = 'pt', padding = "max_length", max_length = self.max_len//2)
        input_for_generation = self.tokenizer(input_for_generation_text, add_special_tokens = True, return_tensors = 'pt', padding = "max_length", max_length = 256)
        
        conclusion_token_ids[0, :conclusion_tokens.shape[1]] = conclusion_tokens
        
        attn_mask[0, :input_tokens.shape[1]] = 1

        return input_token_ids, attn_mask, label_token_ids, input_for_generation, conclusion_token_ids


def collate_fn(batch) :
    input_ids = []
    attn_mask = []
    label_ids = []
    input_for_generation_ids =[]
    input_for_generation_attn_mask = []
    conclusion_ids =[]
    
    for input_token_ids, attn_token_mask, label_token_ids, input_for_generation, conclusion_token_ids in batch:
                
        input_ids.append(input_token_ids)
        attn_mask.append(attn_token_mask)
        label_ids.append(label_token_ids)
        input_for_generation_ids.append(input_for_generation["input_ids"])
        input_for_generation_attn_mask.append(input_for_generation["attention_mask"])
        conclusion_ids.append(conclusion_token_ids)
    
    input_for_generation = {
        "input_ids": torch.cat(input_for_generation_ids, dim = 0),
        "attention_mask": torch.cat(input_for_generation_attn_mask, dim = 0)
    }
    model_inputs = {
        "input_ids" : torch.cat(input_ids, dim = 0),
        "attention_mask" : torch.cat(attn_mask, dim = 0),
        "labels" : torch.cat(label_ids, dim = 0),
        "input_for_generation" : input_for_generation,
        "conclusion" : torch.cat(conclusion_ids, dim = 0)
    }

    return model_inputs