
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder
import torch.nn.functional as F


class Transformer(nn.Module):

    def __init__(self, model, config, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               model = model,
                               config = config,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               model=model,
                               config=config,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    # # truyền vào đây
    # def forward(self, src, trg):
    #     src_mask = self.make_src_mask(src)
    #     trg_mask = self.make_trg_mask(trg)
    #     enc_src = self.encoder(src, src_mask)
    #     output = self.decoder(trg, enc_src, trg_mask, src_mask)
    #     # output shape [batch_size, seq_len, dec_voc_size]
    #     return output
 
    # [batch_size, words, 768]
    # x la text con src voi title la 1 kieu vector
    # minh phai bien trc khi bo vao model 
    def forward(self, src, trg, return_attns = False):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        if return_attns:
            enc_output, enc_attns = self.encoder(src, src_mask, return_attns)
            dec_output, dec_attns, enc_dec_attns = self.decoder(trg, enc_src, trg_mask, src_mask, return_attns)
            return dec_output, enc_output, enc_dec_attns
        else:
            enc_src, enc_attetion = self.encoder(src, src_mask)
            output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
            return output

    def make_src_mask(self, src):
        # print(type(src))
        # print(src.shape)
        # print(src)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    

    def decode_one_step(self, dec_seq, dec_pos, src_seq, enc_output):
        
        # run through the decoder
        vocab_logits, dec_attns, enc_dec_attns = self.decoder(dec_seq, dec_pos, src_seq, enc_output, return_attns=True)

        # Pick the last step
        # lay step cuối cùng có thể hiểu là dec_output này là cho mỗi tokens một last step là từng tokens nên đặt tên hàm là decode one step 
        # vocab_logits [batch_size, seq_len, dec_voc_size]
        vocab_logits = vocab_logits[:, -1, :]  # shape: (batch_size ) * dec_voc_size chi lay tokens cuoi cung
        
        # return vocab_logits, dec_output, enc_dec_attns

        p_vocab = F.log_softmax(vocab_logits, dim=1)
        extra_info = None
        return p_vocab, extra_info

    def decode_extra_info(self, src_seq, batch_hyp, extra):
        return None