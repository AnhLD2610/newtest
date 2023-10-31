
from torch import nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # input shape [*,d_moddel] => output shape [*, hidden]
        self.linear1 = nn.Linear(d_model, hidden)   
        # input shape [*,hidden] => output shape [*, d_model]
        self.linear2 = nn.Linear(hidden, d_model)
        # relu return the same shape 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # s shape [batch_size, length , d_model]
        x = self.linear1(x)
        # s shape [batch_size, length , hidden]
        x = self.relu(x)
        # s shape [batch_size, length , hidden]
        x = self.dropout(x)
        # s shape [batch_size, length , hidden]
        x = self.linear2(x)
        # s shape [batch_size, length , d_model]
        return x
