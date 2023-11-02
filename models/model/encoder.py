
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, model, config, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        model = model,
                                        config = config,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask, return_attns=False):
        enc_slf_attn_list = []

        x = self.emb(x)
        # x shape [batch_size, seq_len, d_model]
        for layer in self.layers:
            x, attention = layer(x, src_mask)
            if return_attns:
                enc_slf_attn_list += [attention]
            # x shape [batch_size, seq_len, d_model]
        # return  [batch_size, seq_len, d_model]
        if return_attns:
            return x, enc_slf_attn_list
        else: return x

    