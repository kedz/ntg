import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, input_dropout=0, dropout=0,
                 transpose=True):
        super(Embedding, self).__init__()

        self.vocab_size_ = vocab_size
        self.embedding_size_ = embedding_size
        self.dropout_ = dropout
        self.input_dropout_ = input_dropout
        self.transpose_ = transpose

        self.lookup_ = nn.Embedding(
            vocab_size + 1, embedding_size, padding_idx=0)

    @property
    def transpose(self):
        return self.transpose_

    @property
    def vocab_size(self):
        return self.vocab_size_

    @property
    def embedding_size(self):
        return self.embedding_size_

    @property
    def input_dropout(self):
        return self.input_dropout_

    @property
    def dropout(self):
        return self.dropout_

    def apply_input_dropout(self, sequence):
        if self.input_dropout > 0 and self.training:
            p = sequence.data.new().float()
            p.resize_(sequence.size()).fill_(self.input_dropout)
            mask = Variable(torch.bernoulli(p).byte())
            return sequence.masked_fill(mask, 0)
            
        else:
            return sequence

    def apply_embedding_dropout(self, embeddings):
        if self.dropout > 0 and self.training:
            embeddings = F.dropout(
                embeddings, p=self.dropout, 
                training=self.training, inplace=True)
        return embeddings 

    def forward(self, inputs):
        if self.transpose:
            if inputs.dim() != 2:
                raise Exception("requires 2 dim tensor to tranpose.")
            else:
                inputs = inputs.transpose(1, 0)
        inputs = self.apply_input_dropout(inputs)

        embeddings = self.lookup_(inputs)
        embeddings = self.apply_embedding_dropout(embeddings)
        return embeddings
