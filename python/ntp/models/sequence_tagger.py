import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceTagger(nn.Module):
    def __init__(self, input_module, encoder_module, predictor_module,
                 pad_value=0):
        super(SequenceTagger, self).__init__()

        self.input_module_ = input_module
        self.encoder_module_ = encoder_module
        self.predictor_module_ = predictor_module
        self.pad_value_ = pad_value

    @property
    def pad_value(self):
        return self.pad_value_

    def forward(self, inputs, mask=None):
        sequence, length = inputs
        batch_size = sequence.size(0)
        sequence_size = sequence.size(1)

        if mask is None:
            mask = sequence.eq(self.pad_value)

        logits_mask = mask.view(batch_size, sequence_size, 1).repeat(
            1, 1, self.predictor_module.output_size).transpose(1, 0)

        encoder_input = self.input_module(sequence)
        context_sequence = self.encoder_module.encoder_context(
            encoder_input, length=length)
        context_flat = context_sequence.view(sequence_size * batch_size, -1)

        logits_flat = self.predictor_module(context_flat)
        logits = logits_flat.view(sequence_size, batch_size, -1).masked_fill(
            logits_mask, 0).transpose(1,0).contiguous()
        return logits
        
    @property
    def input_module(self):
        return self.input_module_

    @property
    def encoder_module(self):
        return self.encoder_module_

    @property
    def predictor_module(self):
        return self.predictor_module_



