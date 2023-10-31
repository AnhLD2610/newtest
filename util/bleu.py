import math
from collections import Counter

import numpy as np


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # print(hypotheses) list of tokens 
    # print(reference) list of tokens
    # print(type(hypotheses)) list
    # print(type(reference)) list
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)

    
def idx_to_word(x, tokenizer):
    # print(x.shape) torch.Size([768]) max_length [1,2,45436,56,345]
    x = x.tolist()
    words = tokenizer.decode(x)
    # print(len(x))
    # print(x.device)
    # print(x.shape)
    # print(x)
    # prediction_ids = unix_coder.generate(x)
    # predictions = unix_coder.decode(prediction_ids)
    # return predictions[0][0]

    # for i in x:
    #     # print(i)
    #     # print(type(i))
    #     word = tokenizer.convert_ids_to_tokens(i)
    #     # print('gggggg')
    #     # print(word)
    #     if '<' not in word:
    #         if 'Ġ' not in word:
    #             words.append(word)
    #         else:
    #             words.append(word[1:])
    # words = " ".join(words)
    
    # words = words.replace('<s><encoder-only></s>','')
    # words = words.replace('</s>','')
    # print(words)
    words = words.replace('<s><encoder-only></s>','')
    search_string = "</s>"  
    start_index = words.find(search_string)  

    if start_index != -1:  
        words = words[:start_index]  
    # else:
    #     print(words)  
    words = words.strip()
    # print(words)
    return words





# def idx_to_word(x, tokenizer):
#     x = x.tolist()
#     words = tokenizer.decode(x)
    
#     # print(x.device)
#     # print(x.shape)
#     # print(x)
#     # prediction_ids = unix_coder.generate(x)
#     # predictions = unix_coder.decode(prediction_ids)
#     # return predictions[0][0]

#     # for i in x:
#     #     # print(i)
#     #     # print(type(i))
#     #     word = tokenizer.convert_ids_to_tokens(i)
#     #     # print('gggggg')
#     #     # print(word)
#     #     if '<' not in word:
#     #         if 'Ġ' not in word:
#     #             words.append(word)
#     #         else:
#     #             words.append(word[1:])
#     # words = " ".join(words)
    
#     # words = words.replace('<s><encoder-only></s>','')
#     # words = words.replace('</s>','')
#     # print(words)
#     words = words.replace('<s><encoder-only></s>','')
#     search_string = "</s>"  
#     start_index = words.find(search_string)  

#     if start_index != -1:  
#         words = words[:start_index]  
#     else:
#         print(words)  
#     words = words.strip()
#     print(words)
#     return words
















# from rouge import FilesRouge
# hyp_path = '/home/aiotlab3/RISE/Lab-MA/DucAnh/transformer_copy/result/hypotheses.txt'
# ref_path = '/home/aiotlab3/RISE/Lab-MA/DucAnh/transformer_copy/result/reference.txt'
# files_rouge = FilesRouge()
# scores = files_rouge.get_scores(hyp_path, ref_path)
# # or
# scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
# print(scores)

