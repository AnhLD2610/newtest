import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
from torchtext.vocab import build_vocab_from_iterator
import json

eng = spacy.load("en_core_web_sm") # Load the English model to tokenize English text
de = spacy.load("en_core_web_sm") # Load the German model to tokenize German text

FILE_PATH = '/home/aiotlab3/RISE/Lab-MA/DucAnh/archive/java.both.val.jsonl'


data_tuples = []

with open(FILE_PATH, "r") as file:
    for line in file:
        data = json.loads(line)
        data_tuple = tuple(data.values())
        data_tuples.append(data_tuple)

data_pipe = []
for t in data_tuples:
    t0 = ' '.join(t[0])
    t1 = ' '.join(t[1])
    new_tuple = (t0, t1)
    data_pipe.append(new_tuple)

def removeAttribution(row):
    """
    Function to keep the first two elements in a tuple
    """
    return row[:2]
data_pipe = data_pipe.map(removeAttribution)