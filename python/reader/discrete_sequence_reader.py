import torch
from reader.reader_base import ReaderBase
from preprocessor import TextPreprocessor
from vocab import Vocab

class DiscreteSequenceReader(ReaderBase):
    def __init__(self, field=0, strip=True, lowercase=True, 
                 replace_digits=True, tokenizer=None,
                 unknown_token="_UNK_", special_tokens=None,
                 top_k=10000000, at_least=1, left_pad=None, right_pad=None):

        if isinstance(special_tokens, str):
            special_tokens = [special_tokens]
        elif special_tokens is None:
            special_tokens = []
            
        if isinstance(left_pad, str):
            left_pad = [left_pad]
        elif left_pad is None:
            left_pad = []

        if isinstance(right_pad, str):
            right_pad = [right_pad]
        elif right_pad is None:
            right_pad = []

        for token in left_pad + right_pad:
            if token not in special_tokens:
                special_tokens.append(token)

        self.left_pad_ = left_pad
        self.right_pad_ = right_pad

        v = Vocab(
            unknown_token=unknown_token, special_tokens=special_tokens,
            at_least=at_least, top_k=top_k)
        pp = TextPreprocessor(
            strip=strip, lowercase=lowercase, replace_digits=replace_digits, 
            tokenizer=tokenizer)
        
        super(DiscreteSequenceReader, self).__init__(field, pp, v)

        self.register_data("data_")
        self.register_data("length_")
        
    def process(self, string):
        tokens = self.left_pad + self.preprocess(string) + self.right_pad
        indices = [self.vocab.index(token) for token in tokens]
        return indices
    
    def save_data(self, datum):
        self.data_.append(datum)
        self.length_.append(len(datum))

    def info(self):
        total = sum(v for k, v in self.vocab.count.items())
        unique = len(self.vocab.count)
        msg = "DiscreteSequenceReader found {} tokens with " \
            "{} unique labels.\n".format(total, unique)
        
        msg += "After pruning, vocabulary has {} unique tokens.\n".format(
            self.vocab.size)

        for i in range(1, min(self.vocab.size, 21)):
            token = self.vocab.token(i)
            count = self.vocab.count.get(token, 0)
            msg += "{}) {} ({})\n".format(i, token, count)
        if i < self.vocab.size:
            msg += ":\n:\n:\n"
        for i in range(self.vocab.size - 20, self.vocab.size):
            token = self.vocab.token(i)
            count = self.vocab.count.get(token, 0)
            msg += "{}) {} ({})\n".format(i, token, count)

        return msg
                
    def finish(self, reset=True):

        data_size = len(self.length_)
        max_len = max(self.length_)
        zed = tuple([0])

        for i in range(data_size):
            if self.length_[i] < max_len:
                self.data_[i] += zed * (max_len - self.length_[i])

        data = torch.LongTensor(self.data_)
        length = torch.LongTensor(self.length_)
        finshed_data = (data, length)

        if reset:
            self.reset()

        return finshed_data

    @property
    def left_pad(self):
        return self.left_pad_

    @property
    def right_pad(self):
        return self.right_pad_
