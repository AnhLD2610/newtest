
import spacy
from collections import Counter


class Tokenizer:

    def __init__(self):
        self.spacy_en = spacy.load('en_core_web_sm')


    def tokenize(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
    
# # spacy_en = spacy.load('en_core_web_sm')
# # # text = 'avsv'
# # text = "Update - The answer is apparently that DbLinq does n't implement Dispose ( ) properly . D'oh ! The below is all sort of misleading - Bottom line : DbLinq is not ( yet ) equivalent to LinqToSql , as I assumed when I originally asked this question . Use it with caution ! I 'm using the Repository Pattern with DbLinq . My repository objects implement IDisposable , and the Dispose ( ) method does only thing -- calls Dispose ( ) on the DataContext . Whenever I use a repository , I wrap it in a using block , like this : This method returns an IEnumerable < Person > , so if my understanding is correct , no querying of the database actually takes place until Enumerable < Person > is traversed ( e.g. , by converting it to a list or array or by using it in a foreach loop ) , as in this example : In this example , Dispose ( ) gets called immediately after setting persons , which is an IEnumerable < Person > , and that 's the only time it gets called.So , three questions : How does this work ? How can a disposed DataContext still query the database for results after the DataContext has been disposed ? What does Dispose ( ) actually do ? I 've heard that it is not necessary ( e.g. , see this question ) to dispose of a DataContext , but my impression was that it 's not a bad idea . Is there any reason not to dispose of a DbLinq DataContext ?","public IEnumerable < Person > SelectPersons ( ) { using ( var repository = _repositorySource.GetPersonRepository ( ) ) { return repository.GetAll ( ) ; // returns DataContext.Person as an IQueryable < Person > } } var persons = gateway.SelectPersons ( ) ; // Dispose ( ) is fired herevar personViewModels = ( from b in persons select new PersonViewModel { Id = b.Id , Name = b.Name , Age = b.Age , OrdersCount = b.Order.Count ( ) } ) .ToList ( ) ; // executes queries"
# # for tok in spacy_en.tokenizer(text):
# #     print(tok)

# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                               RobertaConfig, RobertaModel, RobertaTokenizer)

# model_path = "microsoft/unixcoder-base-nine"
# tokenizer = RobertaTokenizer.from_pretrained(model_path)
# config = RobertaConfig.from_pretrained(model_path)
# import spacy

# # func = "def f(a,b): if a>b: return a else return b"
# # tokens_ids = tokenizer([func],max_length=10,mode="<encoder-only>")

# # print(tokens_ids)

# # Load the English tokenizer
# spacy_en = spacy.load("en_core_web_sm")

# text = "Update - The answer is apparently that DbLinq does n't implement Dispose ( ) properly . D'oh ! The below is all sort of misleading - Bottom line : DbLinq is not ( yet ) equivalent to LinqToSql , as I assumed when I originally asked this question . Use it with caution ! I 'm using the Repository Pattern with DbLinq . My repository objects implement IDisposable , and the Dispose ( ) method does only thing -- calls Dispose ( ) on the DataContext . Whenever I use a repository , I wrap it in a using block , like this : This method returns an IEnumerable < Person > , so if my understanding is correct , no querying of the database actually takes place until Enumerable < Person > is traversed ( e.g. , by converting it to a list or array or by using it in a foreach loop ) , as in this example : In this example , Dispose ( ) gets called immediately after setting persons , which is an IEnumerable < Person > , and that 's the only time it gets called.So , three questions : How does this work ? How can a disposed DataContext still query the database for results after the DataContext has been disposed ? What does Dispose ( ) actually do ? I 've heard that it is not necessary ( e.g. , see this question ) to dispose of a DataContext , but my impression was that it 's not a bad idea . Is there any reason not to dispose of a DbLinq DataContext ? public IEnumerable < Person > SelectPersons ( ) { using ( var repository = _repositorySource.GetPersonRepository ( ) ) { return repository.GetAll ( ) ; // returns DataContext.Person as an IQueryable < Person > } } var persons = gateway.SelectPersons ( ) ; // Dispose ( ) is fired herevar personViewModels = ( from b in persons select new PersonViewModel { Id = b.Id , Name = b.Name , Age = b.Age , OrdersCount = b.Order.Count ( ) } ) .ToList ( ) ; // executes queries"
# i = 0
# x = []
# for tok in spacy_en.tokenizer(text):
    
#     if tok.text not in tokenizer.get_vocab().keys():
#         x.append(tok.text)
#     # print(type(tok))
#     # print(tok.text)
#     i += 1
# # print(i)

# # print(len(tokenizer))  # 28996
# # tokenizer.add_tokens(["NEW_TOKEN1"])
# # tokenizer.add_tokens(["AST#element_binding_expression#Left"])

# # print(len(tokenizer))  # 28997
# # print(config.vocab_size)
# # print(tokenizer.convert_tokens_to_ids("NEW_TOKEN1"))
# # print(type(tokenizer.get_vocab()))
# # # print(tokenizer.get_vocab()[0])
# # # if "NEW_TOKEN1" in tokenizer.get_vocab()[0]:
# # #     print('yes')
# # # for tokenin tokenizer.get_vocab().items():
# # a = tokenizer.get_vocab().keys()

# word_freq = Counter(x)
# word_freq = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)

# for word, frequency in word_freq:
#     print(f"{word}: {frequency}")


# # b = set()
# # for tok in spacy_en.tokenizer(text):
# #     if tok.text not in a:
# #         b.add(tok.text)
# # print(len(b))
# # print(b)
# # if "NEW_TOKEN1" in tokenizer.get_vocab().keys():
# #     print('yes')
# # for token, token_id in vocab_dict.items()[0]:

# # model.resize_token_embeddings(len(tokenizer)) 
# # # The new vector is added at the end of the embedding matrix

# # print(model.embeddings.word_embeddings.weight[-1, :])
# # # Randomly generated matrix

# # model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.config.hidden_size])

# # print(model.embeddings.word_embeddings.weight[-1, :])

















# # outputs a vector of zeros of shape [768]

# # from transformers import T5Model, T5Tokenizer
# # # from transformers import AutoTokenizer

# # model = T5Model.from_pretrained("t5-base")
# # tok = T5Tokenizer.from_pretrained("t5-base",model_max_length=512)

# # enc = tok("some text.", return_tensors="pt")

# # input_ids=enc["input_ids"], 
# # print(input_ids),
# # output = model.encoder(
# #     input_ids=enc["input_ids"], 
# #     attention_mask=enc["attention_mask"], 
# #     return_dict=True
# # )
# # # get the final hidden states
# # emb = output.last_hidden_state
# # print(emb)
# # print(emb.shape)


# # from transformers import T5Tokenizer, T5ForConditionalGeneration

# # tokenizer = T5Tokenizer.from_pretrained("t5-base")
# # # model = T5ForConditionalGeneration.from_pretrained("t5-small")

# # # input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
# # # input_ids = tokenizer.vocab_size("The <extra_id_0> walks in <extra_id_1> park")
# # input_ids = tokenizer.get_vocab()


# # # labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids

# # print(input_ids)
# # # the forward function automatically creates the correct decoder_input_ids
# # # loss = model(input_ids=input_ids, labels=labels).loss
# # # loss.item()