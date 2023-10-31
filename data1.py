import torch
from unixcoder import UniXcoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base-nine")
model.to(device)

context = """
def f(data,file_path):
    # write json data into file_path in python language 2x2 njh_afds-dsaf
"""
# sequence = [0, 6, 2, 1554, 482, 193, 132, 1808, 581, 10241, 3566, 2659, 2, 1]
# tensor = torch.tensor(sequence)

# # If you want to reshape it to have a specific size, e.g., [1, 142], you can do:
# reshaped_tensor = tensor.reshape(1, -1).to(device)
# # print(reshaped_tensor.shape)


tokens_ids = model.tokenize([context],max_length=512,mode="<decoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)


# print(source_ids.shape)
# prediction_ids = model.generate(reshaped_tensor, decoder_only=True, beam_size=3, max_length=128)
# predictions = model.decode(prediction_ids)
# print(predictions.shape)
print(tokens_ids)

# print(context+predictions[0][0])