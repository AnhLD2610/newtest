"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from util.tokenizer import Tokenizer
import pandas as pd 
import torchtext.transforms as T

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        body_text = row['body']
        code_text = row['code']
        title_text = row['title']
        return body_text, code_text, title_text

class MyDataLoader():

    def __init__(self, tokenize, init_token, eos_token):
        self.tokenize = tokenize
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')
 
    def make_dataset(self, path):
        train_df, valid_df, test_df = self.read_data(path)
        # print('1')
        dataset_train = MyDataset(train_df)
        # print('2')
        dataset_valid = MyDataset(valid_df)
        
        dataset_test = MyDataset(test_df)
        
        return dataset_train, dataset_valid, dataset_test

    def read_data(self, path):
        
        new_path = f'{path}/train.csv'
        train_df = pd.read_csv(new_path)
        new_path = f'{path}/valid.csv'
        valid_df = pd.read_csv(new_path)
        new_path = f'{path}/train.csv'
        test_df = pd.read_csv(new_path)

        return train_df, valid_df, test_df

    def getTokens(self, data_iter, batch_size):
        for data in data_iter:
            # print(data)
            # print(len(data))
            # print(len(data[0]))
            # print(data[0])
            # print(type(data[0]))

            for i in range(batch_size):
                # print(i)
                # print(len(data[i]))
                # data[0][i]
                
                text_to_tokenize = data[0][i].strip() + ' ' + data[1][i].strip() + ' ' + data[2][i].strip()
                tokenized_text = self.tokenize.tokenize(text_to_tokenize)
                # print(tokenized_text)
                yield tokenized_text
        
    # thay 32  = biến batch_size
    def build_vocab(self, train_iter, min_freq):
        source_vocab = build_vocab_from_iterator(
            self.getTokens(train_iter, 32),
            min_freq = min_freq,
            specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
            special_first=True
        )
        
        source_vocab.set_default_index(source_vocab['<unk>'])
        vocab_size = len(source_vocab)

        # print(type(source_vocab))        
        # print(source_vocab.get_itos()[:9])
        # print("Vocabulary size:", vocab_size)
        
        return source_vocab, vocab_size 

    def getTransform(self, vocab):

        text_tranform = T.Sequential(
            ## converts the sentences to indices based on given vocabulary
            T.VocabTransform(vocab=vocab),
            ## Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
            # 1 as seen in previous section
            T.AddToken(1, begin=True),
            ## Add <eos> at beginning of each sentence. 2 because the index for <eos> in vocabulary is
            # 2 as seen in previous section
            T.AddToken(2, begin=False)
        )
        return text_tranform
    
    def make_iter(self, train, valid, test,  batch_size):
        train_iterator = DataLoader(train, batch_size=batch_size, shuffle = True, num_workers = 2, pin_memory = False)
        valid_iterator = DataLoader(valid, batch_size=batch_size, shuffle = True, num_workers = 2, pin_memory = False)
        test_iterator = DataLoader(test, batch_size=batch_size, shuffle = True, num_workers = 2, pin_memory = False)
        print('dataset initializing done')

        return train_iterator, valid_iterator, test_iterator


# tokenizer = Tokenizer()  # Create an instance of the Tokenizer class
# data_loader = MyDataLoader(tokenizer, '<sos>', '<eos>')

# train_set, valid_set, test_set = data_loader.make_dataset('/home/aiotlab3/RISE/Lab-MA/DucAnh/transformer/data/C#')
# train_iter, valid_iter, test_iter = data_loader.make_iter(train_set, valid_set, test_set, batch_size=32)
# data_loader.getTokens(train_iter, 32)
# data_loader.build_vocab(train_iter, 5)




# for data in train_iter:
#     print(type(data[0]))
#     print(data[2][0])
#     print(len(data[0]))
#     print(type(data[2][0]))
#     break

# <class 'tuple'>
# How to merge multiple arrays from ASP.Net Core configuration files ?
# 32
# <class 'str'>

# print(train_iter)
# print(type(train_iter))
# data_loader = DataLoader(ext=('.de', '.en'), tokenize='spacy', init_token='<sos>', eos_token='<eos>')

# train_data, valid_data, test_data = data_loader.make_dataset()
# print(type(train_data))
# print("Train Data:", len(train_data))
# print("Validation Data:", len(valid_data))
# print("Test Data:", len(test_data))


# from torch.utils.data import DataLoader
# def collate_fn(batch):
#     texts, labels = [], []
#     for label, txt in batch:
#         texts.append(txt)
#         labels.append(label)
#     return texts, labels
# dataloader = DataLoader(train, batch_size=8, collate_fn=collate_fn)
# for idx, (texts, labels) in enumerate(dataloader):
#     print(idx, texts, labels)





