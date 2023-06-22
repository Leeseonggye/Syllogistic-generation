import torch
import os
from torch.utils.data import Dataset
from utils import merge_data

import pandas as pd

class CustomSyllogismDatasetMixin(Dataset):
    def __init__(self, args, data_name, tokenizer, kfold_idx = None):
        # if (args.augment_mode) & ("train" in data_name):
        #     self.raw_data = merge_data(args, data_name)
        # else:
        #     data_path = os.path.join(args.data_path, data_name)
        #     self.raw_data = pd.read_csv(data_path, encoding = 'Windows-1252')
        #     self.raw_data["index"] = self.raw_data.index
        
        self.data_path = os.path.join(args.data_path, data_name)

        
        if 'Avicenna' in data_name:
            self.raw_data = pd.read_csv(self.data_path, encoding = 'Windows-1252')
            self.raw_data["index"] = self.raw_data.index
            
        else:
            self.raw_data = pd.read_csv(self.data_path)
            self.raw_data = self.raw_data.drop_duplicates()
            print("duplication 제거")
            
        self.raw_data = self.raw_data[self.raw_data["Syllogistic relation"] == "yes"]
        self.max_len = args.max_len
        self.batch_size = args.batch_size

        self.tokenizer = tokenizer
        self.index = self.raw_data["index"]

        self.prem1 = self.raw_data["Premise 1"].to_list()
        self.prem2 = self.raw_data["Premise 2"].to_list()
        self.label = self.raw_data["Conclusion"].to_list()

        if kfold_idx is not None:
            self.prem1 = [sent for num, sent in zip(self.index, self.prem1) if num in kfold_idx]
            self.prem2 = [sent for num, sent in zip(self.index, self.prem2) if num in kfold_idx]
            self.label = [sent for num, sent in zip(self.index, self.label) if num in kfold_idx]


        assert len(self.prem1) == len(self.prem2) and len(self.prem2) == len(self.label),f"데이터 길이가 다름 \n Premise 1 : {len(self.prem1)} \n Premise 2 : {len(self.prem2)} \n Label : {len(self.label)}"
    
    def __getitem__(self, idx):
        return self.__preprocess(self.input_text[idx])

    def __len__(self):
        return len(self.prem1)

    def __preprocess(self, input_text) :
        encoder_text, decoder_target = input_text

        encoder_token_ids = torch.full((1, self.max_len), fill_value = self.input_pad_id)
        decoder_token_ids = torch.full((1, self.max_len), fill_value = self.input_pad_id)
        decoder_target_ids = torch.full((1, self.max_len), fill_value = self.target_pad_id)

        encoder_attn_mask = torch.zeros((1, self.max_len))
        decoder_attn_mask = torch.zeros((1, self.max_len))

        encoder_tokens = self.tokenizer.encode(encoder_text, add_special_tokens = False, return_tensors = 'pt')
        decoder_tokens = self.tokenizer.encode(decoder_target, add_special_tokens = False, return_tensors = 'pt')
        
        encoder_token_ids[0, :encoder_tokens.shape[1]] = encoder_tokens
        decoder_token_ids[0, :decoder_tokens.shape[1]-1] = decoder_tokens[:, :-1]
        decoder_target_ids[0, :decoder_tokens.shape[1]-1] = decoder_tokens[:, 1:]

        encoder_attn_mask[0, :encoder_tokens.shape[1]] = 1
        decoder_attn_mask[0, :decoder_tokens.shape[1]-1] = 1

        return encoder_token_ids, encoder_attn_mask, decoder_token_ids, decoder_attn_mask, decoder_target_ids

    def __get_special_tokens(self, args) :
        pass
    
    def __set_input_text(self) :
        pass


class CustomSyllogismBARTDataset(CustomSyllogismDatasetMixin) :
    def __init__(self, args, data_name, tokenizer, kfold_idx = None):
        super().__init__(args, data_name, tokenizer, kfold_idx)
        self.__get_special_tokens(args)
        self.__set_input_text(args)

    def __get_special_tokens(self, args) :
        self.bos = self.tokenizer.bos_token
        self.sep = self.tokenizer.sep_token
        self.eos = self.tokenizer.eos_token
        self.mask = self.tokenizer.mask_token

        self.input_pad_id = self.tokenizer.pad_token_id
        self.target_pad_id = -100
    
    def __set_input_text(self, args) : 
        if args.generation_target == 'conclusion':
            encoder_text = [f"{self.bos}" + p1  + f"{self.sep} " + p2 + f"{self.eos}" + self.mask for p1, p2 in zip(self.prem1, self.prem2)]
            decoder_target = [f"{self.eos}" + f"{self.bos}" + l + f"{self.eos}" for l in self.label]
            self.input_text =[(enc, dec) for enc, dec in zip(encoder_text, decoder_target)]
        
        elif args.generation_target == 'prem1':
            encoder_text = [f"{self.bos}" + p2  + f"{self.sep} " + label + f"{self.eos}" + self.mask for p2, label in zip(self.prem2, self.label)]
            decoder_target = [f"{self.eos}" + f"{self.bos}" + p1 + f"{self.eos}" for p1 in self.prem1]
            self.input_text =[(enc, dec) for enc, dec in zip(encoder_text, decoder_target)]
        
        elif args.generation_target == 'prem2':
            encoder_text = [f"{self.bos}" + p1  + f"{self.sep} " + label + f"{self.eos}" + self.mask for p1, label in zip(self.prem1, self.label)]
            decoder_target = [f"{self.eos}" + f"{self.bos}" + p2 + f"{self.eos}" for p2 in self.prem2]
            self.input_text =[(enc, dec) for enc, dec in zip(encoder_text, decoder_target)]
            


class CustomSyllogismT5Dataset(CustomSyllogismDatasetMixin) : 
    def __init__(self, args, data_name, tokenizer, kfold_idx = None):
        super().__init__(args, data_name, tokenizer, kfold_idx)
        self.args = args
        self.__get_special_tokens(args)
        self.__set_input_text(args)

    def __get_special_tokens(self, args) :
        self.sep = self.tokenizer.sep_token
        self.eos = self.tokenizer.eos_token

        self.input_pad_id = self.tokenizer.pad_token_id
        self.target_pad_id = -100

    def __set_input_text(self, args):
        prefix = "Syllogistic: "
        if args.generation_target == 'conclusion':
            inputs = [prefix + p1 + " <sep> " + p2 for p1, p2 in zip(self.prem1, self.prem2)]
            self.input_text = [(enc, dec) for enc, dec in zip(inputs, self.label)]
        
        elif args.generation_target == 'prem1':
            inputs = [prefix + p2 + " <sep> " + label for p2, label in zip(self.prem2, self.label)]
            self.input_text = [(enc, dec) for enc, dec in zip(inputs, self.prem1)]
        
        elif args.generation_target == 'prem2':
            inputs = [prefix + p1 + " <sep> " + label for p1, label in zip(self.prem1, self.label)]
            self.input_text = [(enc, dec) for enc, dec in zip(inputs, self.prem2)]
        
        else:
            print("You can generate conclusion or premise. Check your input please.")
    
    def __getitem__(self, idx):
        return self.__preprocess(self.input_text[idx])    

    def __preprocess(self, input_text) :
        encoder_text, decoder_target = input_text
        encoder_inputs = self.tokenizer(encoder_text, max_length = self.args.max_len, padding = "max_length", truncation = True, return_tensors = 'pt')
        decoder_inputs = self.tokenizer(decoder_target, max_length = self.args.max_len, padding = "max_length", truncation = True, return_tensors = 'pt')
        decoder_input_ids = decoder_inputs["input_ids"]
        decoder_input_ids = torch.where(decoder_input_ids != self.input_pad_id, decoder_input_ids, self.target_pad_id)

        return encoder_inputs["input_ids"], encoder_inputs["attention_mask"], decoder_input_ids
    
def collate_fn_bart(batch) :
    encoder_ids = []
    encoder_attn = []
    
    decoder_ids = []
    decoder_attn = []
    decoder_targets = []
    
    for encoder_token_ids, encoder_attn_mask, decoder_token_ids, decoder_attn_mask, decoder_target_ids in batch:
        encoder_ids.append(encoder_token_ids)
        encoder_attn.append(encoder_attn_mask)
        decoder_ids.append(decoder_token_ids)
        decoder_attn.append(decoder_attn_mask)
        decoder_targets.append(decoder_target_ids)

    model_inputs = {
        "input_ids" : torch.cat(encoder_ids, dim = 0),
        "attention_mask" : torch.cat(encoder_attn, dim = 0),
        "decoder_input_ids" : torch.cat(decoder_ids, dim = 0),
        "decoder_attention_mask" : torch.cat(decoder_attn, dim = 0),
        "labels" : torch.cat(decoder_targets, dim = 0)
    }

    return model_inputs

def collate_fn_t5(batch) :
    encoder_ids = []
    encoder_attn_mask = []
    decoder_targets = []

    for encoder_inputs, attn_mask, decoder_inputs in batch:
        encoder_ids.append(encoder_inputs)
        encoder_attn_mask.append(attn_mask)
        decoder_targets.append(decoder_inputs)

    model_inputs = {
        "input_ids" : torch.cat(encoder_ids, dim = 0),
        "attention_mask" : torch.cat(encoder_attn_mask, dim = 0),
        "labels" : torch.cat(decoder_targets, dim = 0)
    }

    return model_inputs