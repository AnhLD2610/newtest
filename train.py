import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from unixcoder1 import UniXcoder
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)

# model_path = "microsoft/unixcoder-base-nine"
# tokenizer = RobertaTokenizer.from_pretrained(model_path)
# config = RobertaConfig.from_pretrained(model_path)
# emb_model = RobertaModel.from_pretrained(model_path) 
# emb_model.to(device)

# unix_coder = UniXcoder(emb_model, config, tokenizer)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = Transformer(model=emb_model,
                    config=config,
                    src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
print(model)
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        # print(batch)
        # print(type(batch))
        src = batch[0].to(device)
        trg = batch[1].to(device)
        # print(src.shape)
        # print(trg.shape)
        # print(src[0])
        # print(trg[1])
    
        # batch.src va batch.trg la câu dạng mà mỗi từ là vị trí trong từ điển 
        # print(src)
        # print(trg)
        optimizer.zero_grad()
        # xem format cua trg 
        output, attn_dist = model(src, trg)
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg.contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        
        # COVERAGE 
        # attention shape [batch_size,n_head,trg_len,src_len] sau khi sum thi sum tat ca cac head_ lai
        attn_dist = torch.sum(attn_dist, dim=1) # [batch_size,trg_len,src_len]
        target_len = attn_dist.shape[1]
        src_len = attn_dist.shape[2]
        
        attn_dist = torch.nn.functional.softmax(attn_dist, dim=2)


        attn_dist_reshaped = attn_dist.contiguous().view(-1, target_len, src_len)
        coverage_vecs = torch.cumsum(attn_dist_reshaped[:, :target_len , :], 1)
        # attn_vecs = attn_dist_reshaped[:, 1:, :]
        # de 3 token dac biet o dau
        attn_vecs = attn_dist_reshaped[:, :, :]

        min_vecs = torch.min(coverage_vecs, attn_vecs)
        coverage_loss = torch.sum(min_vecs).item()
            
        loss += coverage_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            
            # print(src.shape)
            # print(trg.shape)
            # print(src[0])
            # print(trg[1])
        
            # batch.src va batch.trg la câu dạng mà mỗi từ là vị trí trong từ điển 
            # print(src)
            # print(trg)
            optimizer.zero_grad()
            # xem format cua trg 
            output = model(src, trg)
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_new = trg.contiguous().view(-1)
            # print(trg.shape)
            loss = criterion(output_reshape, trg_new)
            
            # COVERAGE 
            # attention shape [batch_size,n_head,trg_len,src_len] sau khi sum thi sum tat ca cac head_ lai
            attn_dist = torch.sum(attn_dist, dim=1) # [batch_size,trg_len,src_len]
            target_len = attn_dist.shape[1]
            src_len = attn_dist.shape[2]

            attn_dist = torch.nn.functional.softmax(attn_dist, dim=2)

            attn_dist_reshaped = attn_dist.contiguous().view(-1, target_len, src_len)
            coverage_vecs = torch.cumsum(attn_dist_reshaped[:, :target_len, :], 1)
            # attn_vecs = attn_dist_reshaped[:, 1:, :]
            # de 3 token dac biet o dau
            attn_vecs = attn_dist_reshaped[:, :, :]

            min_vecs = torch.min(coverage_vecs, attn_vecs)
            coverage_loss = torch.sum(min_vecs).item()
            
            
            epoch_loss += (loss.item()+coverage_loss)

            total_bleu = []
            for j in range(batch_size):
                try:
                    # print('11')
                    # print(trg[j].shape)
                    trg_words = idx_to_word(trg[j], tokenizer)
                    # print('222')
                    output_words = output[j].max(dim=1)[1]
                    # print('333')
                    output_words = idx_to_word(output_words, tokenizer)
                    # print('444')
                    # print(output_words.split(),trg_words.split())
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                    # print(total_bleu)
                    # print('22222')
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


def generate(model, iterator):
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            
            # print(src.shape)
            # print(trg.shape)
            # print(src[0])
            # print(trg[1])
        
            # batch.src va batch.trg la câu dạng mà mỗi từ là vị trí trong từ điển 
            # print(src)
            # print(trg)
            optimizer.zero_grad()
            # xem format cua trg 
            output, attention = model(src, trg)
            # print(output.device)
            # output_reshape = output.contiguous().view(-1, output.shape[-1])
            # trg_new = trg.contiguous().view(-1)
            # print(trg.shape)
            # loss = criterion(output_reshape, trg_new)
            # epoch_loss += loss.item()
            i = 0
            for j in range(batch_size):
                print(i)
                i += 1
                # try:
                # print('11')
                # print(trg[j].shape)
                # trg_words = idx_to_word(trg[j], tokenizer)
                trg_words = idx_to_word(trg[j], tokenizer)
                # print('1323434')
                # print('222')
                output_words = output[j].max(dim=1)[1]
                # print('333')
                # output_words = idx_to_word(output_words, tokenizer)
                output_words = idx_to_word(output_words, tokenizer)

                # print('444')
                with open('result/hypotheses.txt','a', encoding='utf-8') as f1, open('result/reference.txt','a', encoding='utf-8') as f2:
                    f1.write(output_words+'\n')
                    f2.write(trg_words+'\n')
                # print(output_words.split(),trg_words.split())
                # bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                
                # total_bleu.append(bleu)
                # print(total_bleu)
                # print('22222')
                # except:
                #     pass

            
    return 1




if __name__ == '__main__':

    # checkpoint = torch.load("/home/aiotlab3/RISE/Lab-MA/DucAnh/transformer/saved/model-2.499609770909945.pt", map_location=device)
    # # model-1.6922560892358423.pt
    # model.load_state_dict(checkpoint)

    # model.eval()
    # generate(model,test_iter)
    # /home/aiotlab3/RISE/Lab-MA/DucAnh/transformer/saved/model-2.561770428555452.pt
    
    
    # checkpoint = torch.load("/home/aiotlab3/RISE/Lab-MA/DucAnh/transformer_copy/saved/model-1.3572942430148027.pt", map_location=device)
    # # model-1.6922560892358423.pt
    # model.load_state_dict(checkpoint)

    # model.eval()
    
    

    # generate(model,test_iter)
    run(total_epoch=epoch, best_loss=inf)
