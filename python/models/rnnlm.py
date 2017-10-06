from models.sequence_predictor import SequencePredictor
from decoder import RNNDecoder
import torch
import torch.nn as nn

class RNNLM(SequencePredictor):

    @classmethod
    def from_args(cls, args, input_modules, dropout=None,
                  rnn_type=None, target_vocab_size=None,
                  learn_init=None):

        if learn_init is None:
            learn_init = bool(args.learn_init)

        decoder = RNNDecoder.from_args(
            args, decoder_input_size=input_modules.embedding_size, 
            dropout=dropout, rnn_type=rnn_type,
            target_vocab_size=target_vocab_size,
            attention_type="none")

        return cls(input_modules, decoder, learn_init=learn_init)
        
    def __init__(self, input_modules, decoder, learn_init=False):

        super(RNNLM, self).__init__(input_modules, decoder)

        if learn_init:
            init_state = [nn.Parameter(torch.FloatTensor(*dims).fill_(0))
                          for dims in decoder.rnn_state_dims]
            self.init_state = nn.ParameterList(init_state)
        else:
            self.init_state = nn.ParameterList()

    def get_init_state(self, batch_size):
        if len(self.init_state) == 0:
            return None
        elif len(self.init_state) == 1:
            state = self.init_state[0].repeat(1, batch_size, 1)
            return state
        else:
            return tuple([state.repeat(1, batch_size, 1) 
                          for state in self.init_state])

    def forward(self, batch, mask_output=True):
        
        batch_size = batch.decoder_inputs[0].data.size(0)

        init_state = self.get_init_state(batch_size)

        logits = super(RNNLM, self).forward(
            batch.decoder_inputs, prev_state=init_state)

        if mask_output:
            max_steps = batch.decoder_inputs[0].data.size(1)
            mask = batch.decoder_inputs[0].data.t().eq(0)
            mask3d = mask.view(max_steps, batch_size, 1).expand(
                max_steps, batch_size, logits.data.size(2))
            logits.data.masked_fill_(mask3d, 0)
        
        return logits
