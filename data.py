from conf import *
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import csv
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
import spacy 
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, tokenizer, spacy_en, file_path, length, target_len):
        self.examples = []
        self.length = length 
        self.target_len = target_len
        self.spacy_en = spacy_en
        with open(file_path, 'r',encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                body = row['body'].strip()
                code = row['code'].strip()
                title = row['title'].strip()
                self.examples.append((body,code,title))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        tokenizer = self.tokenizer        
        body, code, title = self.examples[i]
        x = []
        for tok in self.spacy_en.tokenizer(body + ' ' + code):
            if tok.text not in tokenizer.get_vocab().keys():
                x.append(tok.text)
        
        word_freq = Counter(x)
        word_freq = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
        special_tokens = []
        for word, frequency in word_freq:
            special_tokens.append(word)
        specical_words = ' '.join(special_tokens)
        code_tokens = tokenizer.tokenize(str(body)+' <csharp> '+str(code))[:self.length-128-4]
        special_tokens = tokenizer.tokenize(specical_words)[:127]
        
        code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]+special_tokens+[tokenizer.sep_token]
        # print(code_tokens)
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        # print(tokenizer.decode(code_ids))
        # code_tokens = tokenizer.convert_ids_to_tokens(code_ids)
        # print(code_tokens)
        padding_length = self.length - len(code_ids)
        code_ids += [tokenizer.pad_token_id]*padding_length
            
            
        title_tokens = tokenizer.tokenize(str(title)) [:self.target_len-4]
        title_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+title_tokens+[tokenizer.sep_token]
        title_ids = tokenizer.convert_tokens_to_ids(title_tokens)
        padding_length = self.target_len - len(title_ids)
        
        title_ids += [tokenizer.pad_token_id]*padding_length
        
        return torch.tensor(code_ids), torch.tensor(title_ids)

model_path = "microsoft/unixcoder-base-nine"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
config = RobertaConfig.from_pretrained(model_path)
emb_model = RobertaModel.from_pretrained(model_path) 
emb_model.to(device)

spacy_en = spacy.load("en_core_web_sm")

dataset_train = TextDataset(tokenizer, spacy_en, train_path, max_len, target_len)
sampler = SequentialSampler(dataset_train)
train_iter = DataLoader(dataset_train, sampler=sampler, batch_size=batch_size,num_workers=4)

dataset_valid = TextDataset(tokenizer, spacy_en, val_path, max_len, target_len)
sampler = SequentialSampler(dataset_valid)
valid_iter = DataLoader(dataset_valid, sampler=sampler, batch_size=batch_size,num_workers=4)

dataset_test = TextDataset(tokenizer, spacy_en, test_path, max_len, target_len)
sampler = SequentialSampler(dataset_test)
test_iter = DataLoader(dataset_test, sampler=sampler, batch_size=batch_size,num_workers=4)




src_pad_idx = tokenizer.convert_tokens_to_ids('<pad>')
trg_pad_idx = tokenizer.convert_tokens_to_ids('<pad>')
trg_sos_idx = tokenizer.convert_tokens_to_ids('<s>')


enc_voc_size = config.vocab_size
dec_voc_size = config.vocab_size 


