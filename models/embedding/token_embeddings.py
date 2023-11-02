import torch 

class TokenEmbedding():
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
    # co the phai to device
    def embedding(self, x):
        source_ids = x.to(self.device)

        # tokens_embeddings,max_func_embedding = model(source_ids)

        with torch.no_grad():
            mask = source_ids.ne(self.config.pad_token_id)
            tokens_embeddings = self.model(source_ids,attention_mask = mask.unsqueeze(1) * mask.unsqueeze(2))[0]
        # [batch_size,512,768]
        # print(tokens_embeddings.shape)
        return tokens_embeddings



























  

