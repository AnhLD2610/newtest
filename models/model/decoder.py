
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, model, config, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        model = model,
                                        config = config,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        trg = self.emb(trg)

        for layer in self.layers:
            trg, dec_slf_attn, dec_enc_attn = layer(trg, enc_src, trg_mask, src_mask)
        # trn shape [batch_size, seq_len, d_model]
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]
        # pass to LM head
        output = self.linear(trg)
        # trn shape [batch_size, seq_len, dec_voc_size]
        if return_attns:
            return output, dec_slf_attn_list, dec_enc_attn_list
        else: return output
