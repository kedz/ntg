import torch
import torch.nn as nn
from input_module.discrete_sequence import DiscreteSequence

class InputGroup(nn.Module):
    def __init__(self):
        super(InputGroup, self).__init__()
        self.input_mods_ = nn.ModuleList()
        self.embedding_size_ = 0

    @property
    def input_modules(self):
        return self.input_mods_

    def add_discrete_sequence(self, vocab_size, embedding_size, dropout=0.0):
        self.input_modules.append(
            DiscreteSequence(vocab_size, embedding_size, dropout))        
        self.embedding_size_ += self.input_modules[-1].embedding_size

    def forward(self, inputs):
        outputs = []
        for input, module in zip(inputs, self.input_modules):
            outputs.append(module(input))

        if len(outputs) == 0:
            return outputs[0]
        else:
            return torch.cat(outputs, 2)

    @property
    def embedding_size(self):
        return self.embedding_size_
