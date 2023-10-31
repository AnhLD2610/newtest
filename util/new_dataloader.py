# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import sys 
import argparse
import logging
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
import csv
logger = logging.getLogger(__name__)
from torchtext.vocab import build_vocab_from_iterator



class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        self.examples = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                body = row['body']
                code = row['code']
                title = row['title']
                self.examples.append((body,code,title))
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        logger.info("Total number: {}".format(len(self.examples)))   

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        args = self.args
        tokenizer = self.tokenizer        
        body, code, title = self.examples[i]
        
        code_tokens = tokenizer.tokenize(str(body)+' '+str(code))[:args.length-4]
        code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        # print(tokenizer.decode(code_ids))
        # code_tokens = tokenizer.convert_ids_to_tokens(code_ids)
        # print(code_tokens)
        padding_length = args.length - len(code_ids)
        code_ids += [tokenizer.pad_token_id]*padding_length
            
            
        title_tokens = tokenizer.tokenize(str(title))[:args.length-4]
        title_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        title_ids = tokenizer.convert_tokens_to_ids(title_tokens)
        padding_length = args.length - len(title_ids)
        title_ids += [tokenizer.pad_token_id]*padding_length
        
        return torch.tensor(code_ids), torch.tensor(title_ids)

model_path = "microsoft/unixcoder-base-nine"
batch_size = 128 
tokenizer = RobertaTokenizer.from_pretrained(model_path)
config = RobertaConfig.from_pretrained(model_path)
model = RobertaModel.from_pretrained(model_path) 
model.to(device)
    
dataset_train = TextDataset(model, tokenizer, "/home/aiotlab3/RISE/Lab-MA/DucAnh/transformer/data/C#/train.csv")
sampler = SequentialSampler(dataset_train)
dataloader = DataLoader(dataset_train, sampler=sampler, batch_size=batch_size,num_workers=4)

dataset_valid = TextDataset(tokenizer, val_path)
sampler = SequentialSampler(dataset_train)
dataloader = DataLoader(dataset_train, sampler=sampler, batch_size=batch_size,num_workers=4)

dataset_test = TextDataset(tokenizer, test_path)
sampler = SequentialSampler(dataset_train)
dataloader = DataLoader(dataset_train, sampler=sampler, batch_size=batch_size,num_workers=4)

# def getTokens(file_path):
#     with open(file_path, 'r') as csv_file:
#         csv_reader = csv.DictReader(csv_file)
#         for row in csv_reader:
#             body = row['body']
#             code = row['code']
#             title = row['title']
#             line = body.strip() + ' ' + code.strip() + ' ' + title.strip()
#             yield tokenizer.tokenize(line)
#             # yield line.strip().split()


# def build_vocab(file_path, min_freq):
#     source_vocab = build_vocab_from_iterator(
#         getTokens(file_path),
#         min_freq = min_freq,
#         specials= ['<s>', '<pad>', '</s>', '<unk>'],
#         #"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3
#         special_first=True
#     )
#     # source_vocab.set_default_index(source_vocab['<unk>'])
#     vocab_size = len(source_vocab)

#     # print(type(source_vocab))        
#     # print(source_vocab.get_itos()[:9])
#     # print("Vocabulary size:", vocab_size)
#     print(vocab_size)
    # return source_vocab, vocab_size 

# source_vocab, vocab_size  = build_vocab("/home/aiotlab3/RISE/Lab-MA/DucAnh/transformer/data/C#/train.csv",1)
# with open('out.txt','w',encoding='utf-8') as f:
#     for i in range (vocab_size):
#         f.write(source_vocab.get_itos()[i]+'\n')
        
# def main():
#     parser = argparse.ArgumentParser()

#     ## Required parameters
#     parser.add_argument("--output_dir", default="tmp/", type=str)
#     parser.add_argument("--file_path", default="samples.jsonl", type=str)
#     parser.add_argument("--model_path", default="microsoft/unixcoder-base-nine", type=str)
#     parser.add_argument("--length", default=128, type=int)    
#     parser.add_argument("--eval_batch_size", default=64, type=int)
    
#     #print arguments
#     args = parser.parse_args()
#     #set log
#     logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
#     #set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     args.n_gpu = torch.cuda.device_count()
#     args.device = device
#     logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir) 
        
#     #build model
#     tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
#     config = RobertaConfig.from_pretrained(args.model_path)
#     model = RobertaModel.from_pretrained(args.model_path) 
#     logger.info("Training/evaluation parameters %s", args)
    
#     model.to(args.device)
#     if args.n_gpu > 1:
#         model = torch.nn.DataParallel(model)  
#     model.eval()
    
  
    
#     dataset = TextDataset(tokenizer, args)
#     sampler = SequentialSampler(dataset)
#     dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,num_workers=4)



#     code_vecs = [] 
#     for batch in tqdm(dataloader):  
#         source_ids = batch.to(args.device)
#         with torch.no_grad():
#             mask = source_ids.ne(config.pad_token_id)
#             token_embeddings = model(source_ids,attention_mask = mask.unsqueeze(1) * mask.unsqueeze(2))[0]
#             sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
#         code_vecs.append(sentence_embeddings.cpu().numpy().astype(np.float16))
#     code_vecs = np.concatenate(code_vecs,0)
#     dic = {}
#     for x,y in zip(dataset.examples,code_vecs):
#         dic[x[0]] = y
#     pickle.dump(dic,open(args.output_dir+"/feat.pkl","wb"))


# if __name__ == "__main__":
#     main()


