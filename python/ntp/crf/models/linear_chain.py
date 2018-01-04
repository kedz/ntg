import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import LinearChainDecoder

class LinearChain(nn.Module):
    def __init__(self, num_states, feature_extractor):
        super(LinearChain, self).__init__()

        self.decoder_ = LinearChainDecoder(num_states)
        self.feature_extractor_ = feature_extractor


    @property
    def num_states(self):
        return self.decoder.num_states

    @property
    def decoder(self):
        return self.decoder_

    @property
    def feature_extractor(self):
        return self.feature_extractor_


    def get_features(self, inputs):
        return self.feature_extractor(inputs).transpose(1, 0).contiguous()

   

    def score_state(self, inputs, states, normalized=False):
        
        features = self.get_features(inputs)
        scores = self.decoder.score_state_sequence(
            features, states, sequence_sizes=inputs.length)

        if normalized:

            log_normalizer = self.decoder.forward_algorithm(
                features, sequence_sizes=inputs.length)
            return scores - log_normalizer

        else:

            return scores

    def predict(self, inputs):

        features = self.get_features(inputs)
        states = self.decoder.viterbi_decode(
            features, sequence_sizes=inputs.length)
        return states

    def forward(self, inputs, state_sequences=None):

        print("AHHHHHH")
        exit()


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



