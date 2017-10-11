import torch
import torch.nn as nn


class DiscreteFeature(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout):
        
        super(DiscreteFeature, self).__init__()

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

    def forward_sequence(self, input, max_steps):

        emb = self.embedding_lookup(input)
        emb_seq = emb.unsqueeze(0).repeat(max_steps, 1, 1)
        if self.dropout > 0:
            emb_seq = F.dropout(
                emb_seq, p=self.dropout, training=self.training, inplace=True)
        return emb_seq
