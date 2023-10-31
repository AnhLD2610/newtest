
from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        # w_q, w_k, w_v, w_concat transform [*, d_model] to [*, d_model]
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # q, k, v được biến đổi thành q*w_q, k*w_k, v*w_v
        # shape q, k ,v = [batch_size, length, d_model]
            
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        # out = [batch_size, head, length, d_tensor]
        out, attention = self.attention(q, k, v, mask=mask)
        
        print(out.shape)
        print(attention.shape)
        # 4. concat and pass to linear layer
        # out shape [batch_size, length , d_model]
        out = self.concat(out)
        # out shape [batch_size, length , d_model]
        out = self.w_concat(out)


        # combined_weights = w1 * attention_head_1 + w2 * attention_head_2 + ... + wh * attention_head_h
        # attention_distribution = softmax(combined_weights, axis=-1)



        # split và concat ngược lại với nhau, split xong lại concat về ban đầu

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out, attention

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        # d_tensor ddaay
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        # return: [batch_size, length, d_model]
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


import torch

# Define the parameters
d_model = 64  # Dimension of the model
n_head = 4    # Number of attention heads
seq_length = 10
batch_size = 2

# Create an instance of MultiHeadAttention
multi_head_attention = MultiHeadAttention(d_model, n_head)

# Generate random input tensors for q, k, and v
q = torch.randn(batch_size, seq_length, d_model)
k = torch.randn(batch_size, seq_length, d_model)
v = torch.randn(batch_size, seq_length, d_model)

# Call the forward method of MultiHeadAttention
output, attention = multi_head_attention(q, k, v)

# Print the output shape
print("Output shape:", output.shape)
print("attention shape:", attention.shape)

# print(attention)
import torch

# Assuming you have your attention values as attention_tensor
attention_tensor = torch.randn(10, 64)  # Replace this with your actual attention values

# Apply softmax to get the attention distribution
attention_distribution = torch.softmax(attention_tensor, dim=1)

# print(attention_distribution)
# # source_length, emb
# print(attention_distribution.shape)


# time_step, source length
# target_length, source_length 