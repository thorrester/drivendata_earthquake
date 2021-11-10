import pandas as pd
import numpy as np
from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer, Regex, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, StripAccents, Replace
from tokenizers.pre_tokenizers import Whitespace


def convert_to_sentence(x, columns):
    sentence = []
    for col in columns:
        term = f"{col}-{x[col]}"
        sentence.append(term)
    sentence = " ".join(sentence)

    return sentence


class WordpieceTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        self.trainer = WordPieceTrainer(
            vocab_size=30000,
            min_frequency=2,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            show_progress=False,
        )
        self.tokenizer.normalizer = normalizers.Sequence(
            [
                StripAccents(),
                Lowercase(),
            ]
        )
        self.tokenizer.pre_tokenizer = Whitespace()

    def fit(self, data):

        # Sequence length
        self.sequence_length = int(np.quantile([len(sent) for sent in data], 0.75))

        # Enable padding and truncation
        self.tokenizer.enable_padding(length=self.sequence_length)
        self.tokenizer.enable_truncation(max_length=self.sequence_length)

        # fit
        self.tokenizer.train_from_iterator(data, trainer=self.trainer)

    def transform(self, data):

        sentences = self.tokenizer.encode_batch(data)
        x_batch = np.array([sentence.ids for sentence in sentences], dtype="int64")

        return x_batch

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
