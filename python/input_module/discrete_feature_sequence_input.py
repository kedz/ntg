import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteFeatureSequenceInput(nn.Module):

    def __init__(self, vocab_sizes, embedding_sizes):
        super(DiscreteFeatureSequenceInput, self).__init__()

        if isinstance(vocab_sizes, int):
            vocab_sizes = [vocab_sizes]
        if isinstance(embedding_sizes, int):
            embedding_sizes = [embedding_sizes]
        if len(vocab_sizes) != len(embedding_sizes):
            raise Exception(
                "Must have one vocab size for each embedding size.")
        
        self.embedding_sizes_ = tuple(embedding_sizes)
        self.embedding_size_ = sum(embedding_sizes)
        self.vocab_sizes_ = tuple(vocab_sizes)
        self.vocab_size_ = sum(vocab_sizes)
        self.reset_module()        

    def reset_module(self):

        self.embedding_modules_ = nn.ModuleList(
            [nn.Embedding(vs, es, padding_idx=0) 
             for vs, es in zip(self.vocab_sizes, self.embedding_sizes)])
    
    @property
    def embedding_modules(self):
        return self.embedding_modules_

    @property
    def embedding_size(self):
        return self.embedding_size_

    @property
    def embedding_sizes(self):
        return self.embedding_sizes_

    @property
    def vocab_size(self):
        return self.vocab_size_

    @property
    def vocab_sizes(self):
        return self.vocab_sizes_

    def forward(self, inputs, input_length=None, transpose=True):

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        embedded_inputs = []
        for input, lookup in zip(inputs, self.embedding_modules):
            if transpose:
                input = input.t()
            embedded_inputs.append(lookup(input))

        if len(embedded_inputs) > 1:
            input_sequence = torch.cat(embedded_inputs, 2)
        else:
            input_sequence = embedded_inputs[0]

        if input_length is None:
            return input_sequence
        else:
            input_sequence_packed = nn.utils.rnn.pack_padded_sequence(
                input_sequence, input_length.data.tolist(), batch_first=False)
            return input_sequence_packed
