from .field_reader_base import FieldReaderBase
from ..vocabulary import Vocabulary

import sys
import torch
import re


DEFAULT_TOPK = sys.maxsize

class Token(FieldReaderBase):

    def __init__(self, field, unknown_token="__UNK__",
                 top_k=DEFAULT_TOPK, at_least=1,
                 lower=True, replace_digit="#"):
        super(Token, self).__init__(field)
        
        self.unknown_token_ = unknown_token
        self.lower_ = lower
        self.replace_digit_ = replace_digit

        self.register_data("indices")

        self.vocab_ = Vocabulary(
            zero_indexing=False,
            unknown_token=unknown_token, top_k=top_k, at_least=at_least)

    def read_extract(self, data):

        if self.lower:
            data = data.lower()
        if self.replace_digit is not None:
            data = re.sub(r"[0-9]", self.replace_digit, data) 
        index = self.vocab[data]
        if index is None:
            raise Exception(
                "Found unknown token: {}".format(data))
        self.indices.append(index)

    def fit_parameters(self):
        self.vocab.freeze()
    
    def finalize_saved_data(self):
        return (torch.LongTensor(self.indices),)

    @property
    def vocab(self):
        return self.vocab_

    @property
    def unknown_token(self):
        return self.unknown_token_

    @property
    def zero_indexing(self):
        return self.vocab.zero_indexing

    @property
    def lower(self):
        return self.lower_

    @property
    def replace_digit(self):
        return self.replace_digit_

    @property
    def at_least(self):
        return self.at_least_

    @property
    def top_k(self):
        return self.top_k_
