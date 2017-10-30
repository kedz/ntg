import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def set_dropout(self, dropout):
        self.dropout_ = dropout

    @property
    def embedding_lookup(self):
        return self.embedding_lookup_

    def forward_sequence(self, input, max_steps, transpose=True):

        if transpose:
            input = input.t()

        emb = self.embedding_lookup(input)
        
        if self.dropout > 0:
            emb = F.dropout(
                emb, p=self.dropout, training=self.training, inplace=True)
        
        return emb

    def forward_step(self, input, step):
        batch_size = input.size(0)
        emb = self.embedding_lookup(input.contiguous().view(1, batch_size))

        if self.dropout > 0:
            emb = F.dropout(
                emb, p=self.dropout, training=self.training, inplace=True)
        
        return emb
