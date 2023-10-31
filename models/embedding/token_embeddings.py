import torch 
import sys
sys.path.append('/home/aiotlab3/RISE/Lab-MA/DucAnh/transformer')
from conf import *




# class TokenEmbedding(nn.Embedding):
#     """
#     Token Embedding using torch.nn
#     they will dense representation of word using weighted matrix
#     """

#     def __init__(self, vocab_size, d_model):
#         """
#         class for token embedding that included positional information

#         :param vocab_size: size of vocabulary
#         :param d_model: dimensions of model
#         """
#         super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

# # return embedding size (*,d_model)
# # co padding 

# class TokenEmbedding(nn.Embedding):
#     """
#     Token Embedding using torch.nn
#     they will dense representation of word using weighted matrix
#     """

#     # def __init__(self, vocab_size, d_model):
#     #     """
#     #     class for token embedding that included positional information

#     #     :param vocab_size: size of vocabulary
#     #     :param d_model: dimensions of model
#     #     """
#     #     super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
        
class TokenEmbedding():
    def __init__(self, model, config):
        self.model = model
        self.config = config
    # co the phai to device
    def embedding(self, x):
        source_ids = x.to(device)

        # tokens_embeddings,max_func_embedding = model(source_ids)

        with torch.no_grad():
            mask = source_ids.ne(self.config.pad_token_id)
            tokens_embeddings = self.model(source_ids,attention_mask = mask.unsqueeze(1) * mask.unsqueeze(2))[0]
        # [batch_size,512,768]
        # print(tokens_embeddings.shape)
        return tokens_embeddings


# class TextDataset(Dataset):
#     def __init__(self, tokenizer, file_path, length):
#         self.examples = []
#         self.length = length 
#         with open(file_path, 'r') as csv_file:
#             csv_reader = csv.DictReader(csv_file)
#             for row in csv_reader:
#                 body = row['body']
#                 code = row['code']
#                 title = row['title']
#                 self.examples.append((body,code,title))
#         self.model = model
#         self.tokenizer = tokenizer
#         logger.info("Total number: {}".format(len(self.examples)))   

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):   
#         tokenizer = self.tokenizer        
#         body, code, title = self.examples[i]
        
#         code_tokens = tokenizer.tokenize(str(body)+' '+str(code))[:self.length-4]
#         code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
#         code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
#         # print(tokenizer.decode(code_ids))
#         # code_tokens = tokenizer.convert_ids_to_tokens(code_ids)
#         # print(code_tokens)
#         padding_length = self.length - len(code_ids)
#         code_ids += [tokenizer.pad_token_id]*padding_length
            
            
#         title_tokens = tokenizer.tokenize(str(title))[:self.length-4]
#         title_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+title_tokens+[tokenizer.sep_token]
#         title_ids = tokenizer.convert_tokens_to_ids(title_tokens)
#         padding_length = self.length - len(title_ids)
#         title_ids += [tokenizer.pad_token_id]*padding_length
        
#         return torch.tensor(code_ids), torch.tensor(title_ids)

# model_path = "microsoft/unixcoder-base-nine"
# tokenizer = RobertaTokenizer.from_pretrained(model_path)
# config = RobertaConfig.from_pretrained(model_path)
# model = RobertaModel.from_pretrained(model_path) 
# model.to(device)
    
# dataset_train = TextDataset(tokenizer, train_path, 512)
# sampler = SequentialSampler(dataset_train)
# dataloader = DataLoader(dataset_train, sampler=sampler, batch_size=batch_size,num_workers=4)

# dataset_valid = TextDataset(tokenizer, val_path, 512)
# sampler = SequentialSampler(dataset_train)
# dataloader = DataLoader(dataset_train, sampler=sampler, batch_size=batch_size,num_workers=4)

# dataset_test = TextDataset(tokenizer, test_path, 512)
# sampler = SequentialSampler(dataset_train)
# dataloader = DataLoader(dataset_train, sampler=sampler, batch_size=batch_size,num_workers=4)



# for batch in tqdm(dataloader): 
#     # print(type(batch))
#     # batch = torch.tensor(batch)
#     print(type(batch[0]))
#     token = TokenEmbedding(model, config)
#     # print(batch[0])
#     print(batch[0].shape)
#     token.embedding(batch[0])
#     break




























  

# small 60M
# base 220m
# T5 
# from transformers import AutoTokenizer, T5Model

# tokenizer = AutoTokenizer.from_pretrained("t5-base")
# model = T5Model.from_pretrained("t5-base")

# enc = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")

# output = model.encoder(
#     input_ids=enc["input_ids"], 
#     attention_mask=enc["attention_mask"], 
#     return_dict=True
# )
# emb = output.last_hidden_state
# print(emb)
# print(emb.shape)
# print(enc["input_ids"])

# # Tokenize the input sentence
# tokens = tokenizer.tokenize("Studies have been shown that owning a dog is good for you")

# # Print the list of tokens
# print(tokens)


# codet5
# from transformers import AutoModel, AutoTokenizer

# checkpoint = "Salesforce/codet5p-110m-embedding"
# device = "cuda"  # for GPU usage or "cpu" for CPU usage

# tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
# model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

# inputs = tokenizer.encode("def print_hello_world():\tprint('Hello World!')", return_tensors="pt").to(device)
# embedding = model(inputs)[0]
# print(f'Dimension of the embedding: {embedding.size()[0]}, with norm={embedding.norm().item()}')
# # Dimension of the embedding: 256, with norm=1.0
# # print(embedding)
# # print(embedding.shape)
# a = model(inputs)
# print(a)
# print(a.shape)

# 768 
# 256

# import torch
# from unixcoder import UniXcoder

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UniXcoder("microsoft/unixcoder-base-nine")
# model.to(device)


# def count_parameters(model):
#         return sum(p.numel() for p in model.parameters() if not p.requires_grad)
# print(f'The model has {count_parameters(model):,} trainable parameters')
# # # The model has 125,929,728 trainable parameters

# des = [" I have a Custom User model that takes user ip address. I want to add the IP address of the user upon completion of the sign up form. Where do I implement the below code? I am not sure whether to put this into my forms.py or views.py file. I expect to be able to save the user's ip address into my custom user table upon sign up."]
# code = ["def get_client_ip(request): x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR') if x_forwarded_for: ip = x_forwarded_for.split(',')[0] else: ip = request.META.get('REMOTE_ADDR') return ip"]
# concatenated_list = des + code
# tokens_ids = model.tokenize(['','df','dffff'],max_length=1023,mode="<encoder-only>",padding=True)
# # tokens_ids = tokens_ids[:,:512]
# source_ids = torch.tensor(tokens_ids).to(device)
# des_tokens_embeddings,des_embedding = model(source_ids)
# # print(tokens_ids)
# print(des_tokens_embeddings)
# print(des_tokens_embeddings.shape)




# print(des_tokens_embeddings)
# print(des_tokens_embeddings.shape)

# tokens = TokenEmbedding(model)
# a= tokens.embedding(des,code)
# print(a)
# print(a.shape)
# import torch
# from unixcoder import UniXcoder

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UniXcoder("microsoft/unixcoder-base")
# model.to(device)


# func = ["def f(a,b): if a>b: return a else return b",'if a < b']
# # func = "def f(a,b): if a>b: return a else return b"
# tokens_ids = model.tokenize(func,max_length=512,mode="<encoder-only>",padding=True)
# # print(tokens_ids)



# source_ids = torch.tensor(tokens_ids).to(device)



# prediction_ids = model.generate(source_ids, decoder_only=True, beam_size=3, max_length=128)
# predictions = model.decode(prediction_ids)
# print(predictions[1][0])

# tokens_embeddings,max_func_embedding = model(source_ids)



# print(tokens_embeddings.shape)
# print(max_func_embedding.shape)
