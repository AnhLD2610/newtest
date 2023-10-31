
import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor] d_tensor la dimension cua q,k,v
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        # k_t shape [batch_size, head, d_tensor, length]
        k_t = k.transpose(2, 3)  # transpose
        # score shape = [batch_size, head, length, length]
        # score shape = [batch_size, head, target_len, src_len]

        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        #[batch_size, head, length, length] * [batch_size, head, length, d_tensor]
        # v shape = [batch_size, head, length, length]
        v = score @ v
        # v shape = [batch_size, head, length, d_tensor]
        
        return v, score
    
