import torch
import torch.nn as nn


class DiscreteSequence(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout):
        
        super(DiscreteSequence, self).__init__()

        self.vocab_size_ = vocab_size
        self.embedding_size_ = embedding_size
        self.dropout_ = dropout
        
        self.embedding_lookup_ = nn.Embedding(
            vocab_size, embedding_size, padding_idx=0)

    @property
    def vocab_size(self):
        return self.vocab_size_

    @property
    def embedding_size(self):
        return self.embedding_size_

    @property
    def dropout(self):
        return self.dropout_

    @property
    def embedding_lookup(self):
        return self.embedding_lookup_

    def forward(self, input, transpose=True):

        if transpose:
            input = input.t()

        emb = self.embedding_lookup(input)
        
        if self.dropout > 0:
            emb = F.dropout(
                emb, p=self.dropout, training=self.training, inplace=True)
        
        return emb
