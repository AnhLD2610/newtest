
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        # dec shape [batch_size, length, d_model]
        _x = dec
        x, selfAttention = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x, dec_enc_attn = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            # print("Shape of x after enc_dec_attention:", x.shape)

            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        # x shape [batch_size, length , d_model]

        return x, selfAttention, dec_enc_attn 




# import torch


# d_model = 512
# ffn_hidden = 2048
# n_head = 8
# drop_prob = 0.1
# batch_size = 32
# seq_length = 10

# # Create a DecoderLayer instance
# decoder_layer = DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)

# # Generate some example input tensors (you would typically have your own data)
# dec = torch.randn(batch_size, seq_length, d_model)
# enc = torch.randn(batch_size, seq_length, d_model)
# trg_mask = None  # You can define the mask as needed
# src_mask = None  # You can define the mask as needed

# # Call the forward method
# x = decoder_layer(dec, enc, trg_mask, src_mask)

# # Print the shape of x after the enc_dec_attention operation
# print("Shape of x after enc_dec_attention:", x.shape)






