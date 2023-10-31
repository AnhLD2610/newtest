import torch

# GPU device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
batch_size = 4
max_len = 768
# max_len = 1023
# d_model = 512
d_model = 768
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9

patience = 10
warmup = 100
# epoch = 1000
epoch = 6
clip = 1.0
weight_decay = 5e-4
inf = float('inf')

# train, val, path 
train_path ="/kaggle/input/c#_dataset/train.csv"
val_path ="/kaggle/input/c#_dataset/valid.csv"
test_path = "/kaggle/input/c#_dataset/test.csv"

