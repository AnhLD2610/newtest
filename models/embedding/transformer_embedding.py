
from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, model, config, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        # d_model = 768 
        super(TransformerEmbedding, self).__init__()
        # tok_emb shape [batch_size, seq_len, d_model]
        self.tok_emb = TokenEmbedding(model,config, device)
        # pos_emb shape [seq_len, d_model]
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    # x la text
    def forward(self, x):
        # print('5555555555555')
        # # tok_emb shape [batch_size, seq_len, d_model]
        # # print(x)
        # tok_emb = self.tok_emb.embedding(x)
        # print('55555555555556')
        # seq_len = tok_emb.shape[1]
        # print('55555555555557')

        # # pos_emb shape [seq_len, d_model]
        # pos_emb = self.pos_emb(seq_len)
        # print('55555555555585')
        # # tok_emb + pos_emb shape = [batch_size, seq_len, d_model]
        # return self.drop_out(tok_emb + pos_emb)
    
        tok_emb = self.tok_emb.embedding(x)
        pos_emb = self.pos_emb(x)
        # print('ttttttt')
        # print(x.shape)
        # tok_emb shape [batch_size, seq_len, d_model]
        # pos_emb shape [seq_len, d_model]
        # print(tok_emb.shape)
        # print(pos_emb.shape)
        return self.drop_out(tok_emb + pos_emb)
