from .field_reader_base import FieldReaderBase
from ..vocabulary import Vocabulary

import sys
import torch
import re


DEFAULT_TOPK = sys.maxsize

class TokenSequence(FieldReaderBase):

    def __init__(self, field, sep=" ", unknown_token="__UNK__",
                 start_token=None, stop_token=None,
                 top_k=DEFAULT_TOPK, at_least=1,
                 lower=True, replace_digit="#"):
        super(TokenSequence, self).__init__(field)
        
        special_tokens = [token for token in [start_token, stop_token]
                          if token is not None]

        self.start_token_ = start_token
        self.stop_token_ = stop_token
        self.unknown_token_ = unknown_token
        self.lower_ = lower
        self.replace_digit_ = replace_digit
        self.sep_ = sep

        self.register_data("sequences")
        self.register_data("sequence_lengths")

        self.vocab_ = Vocabulary(
            zero_indexing=False, special_tokens=special_tokens,
            unknown_token=unknown_token, top_k=top_k, at_least=at_least)

    def read_extract(self, data):

        if isinstance(data, str):
            if self.sep is not None:
                tokens = data.split(self.sep)
            else:
                tokens = [data]
        else:
            tokens = data

        if self.lower:
            tokens = [token.lower() for token in tokens]
        if self.replace_digit is not None:
            tokens = [re.sub(r"[0-9]", self.replace_digit, token) 
                      for token in tokens]
        if self.start_token:
            tokens = [self.start_token] + tokens
        if self.stop_token:
            tokens = tokens + [self.stop_token]
        indices = [] 
        for token in tokens:
            index = self.vocab[token]
            if index is None:
                raise Exception(
                    "Found unknown token: {}".format(token))
            indices.append(index)

        self.sequences.append(indices)
        self.sequence_lengths.append(len(indices))
        
    def fit_parameters(self):
        self.vocab.freeze()
    
    def finalize_saved_data(self):
        sizes = torch.LongTensor(self.sequence_lengths)
        max_size = sizes.max()
        for seq in self.sequences:
            diff = max_size - len(seq)
            if diff > 0:
                seq += [0] * diff
        sequences = torch.LongTensor(self.sequences)

        return sequences, sizes

    @property
    def vocab(self):
        return self.vocab_

    @property
    def start_token(self):
        return self.start_token_

    @property
    def stop_token(self):
        return self.stop_token_

    @property
    def unknown_token(self):
        return self.unknown_token_

    @property
    def zero_indexing(self):
        return self.vocab.zero_indexing

    @property
    def sep(self):
        return self.sep_

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
