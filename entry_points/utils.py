import pandas as pd
from tokenizers import BertWordPieceTokenizer, normalizers, Regex
from tokenizers.normalizers import Lowercase, StripAccents, Replace
from tokenizers.pre_tokenizers import Whitespace

def convert_to_sentence(x, columns):
    sentence = []
    for col in columns:
        term = f'{col}-{x[col]}'
        sentence.append(term)
    sentence = ' '.join(sentence)

    return sentence

class Tokenizer:
    self.sequence_length