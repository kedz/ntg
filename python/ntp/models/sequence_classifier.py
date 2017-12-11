import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceClassifier(nn.Module):
    def __init__(self, input_module, encoder_module, predictor_module):
        super(SequenceClassifier, self).__init__()

        self.input_module_ = input_module
        self.encoder_module_ = encoder_module
        self.predictor_module_ = predictor_module

    def forward(self, inputs):
        sequence, length = inputs
        encoder_input = self.input_module(sequence)
        encoded_input = self.encoder_module.encoder_state_output(
            encoder_input, length=length)
        output = self.predictor_module(encoded_input)
        return output
        
    @property
    def input_module(self):
        return self.input_module_

    @property
    def encoder_module(self):
        return self.encoder_module_

    @property
    def predictor_module(self):
        return self.predictor_module_



