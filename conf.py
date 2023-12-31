import torch

# GPU device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# batch_size = 4
batch_size = 8

max_len = 768
target_len = 128
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
epoch = 10
# epoch = 6
clip = 1.0
weight_decay = 5e-4
inf = float('inf')

# train, val, path 
train_path ="/media/data/thanhnb/newtest/data/C#/train.csv"
val_path ="/media/data/thanhnb/newtest/data/C#/valid.csv"
test_path = "/media/data/thanhnb/newtest/data/C#/test.csv"

