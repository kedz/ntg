from models.sequence_predictor import SequencePredictor
from decoder import RNNDecoder
import torch
import torch.nn as nn
from torch.autograd import Variable

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

    # TODO make this a mixin, also for rnnseq2seq
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
        inputs = batch.decoder_inputs + batch.decoder_features
        batch_size = batch.decoder_inputs[0].data.size(0)
        max_steps = batch.decoder_inputs[0].data.size(1)

        init_state = self.get_init_state(batch_size)

        logits, state = super(RNNLM, self).forward(
            inputs, max_steps, prev_state=init_state)

        if mask_output:
            mask = batch.decoder_inputs[0].data.t().eq(0)
            mask3d = mask.view(max_steps, batch_size, 1).expand(
                max_steps, batch_size, logits.data.size(2))
            logits.data.masked_fill_(mask3d, 0)
        
        return logits

    def init_beam_rnn_state(self, state, beam_size):
        
        if isinstance(state, type(None)):
            return None
        elif isinstance(state, (tuple, list)):
            return (self.init_beam_rnn_state(state_i, beam_size) 
                    for state_i in state)
        else:
            num_layers = state.size(0)
            batch_size = state.size(1)
            hidden_size = state.size(2)
            return state.contiguous().repeat(1, 1, beam_size).view(
                num_layers, batch_size * beam_size, hidden_size)

    def update_beam_state(self, state, source):
        if isinstance(state, type(None)):
            return None
        elif isinstance(state, (tuple, list)):
            return (self.update_beam_state(state_i, source)
                    for state_i in state)
        else:
            next_state = state.data.new(state.size())
            i = 0
            for batch in range(source.size(0)):
                for beam in range(source.size(1)):
                    loc = batch * source.size(1) + source[batch, beam]
                    next_state[:,i,:].copy_(state.data[:, loc])
                    i += 1
            return Variable(next_state)

    def complete_sequence(self, decoder_inputs, decoder_features=None, 
            max_steps=100, beam_size=8):

        batch_size = decoder_inputs[0].size(0)   
        prefix_size = decoder_inputs[0].size(1) - 1  

        if decoder_features is None:
            decoder_features = []

        inputs = decoder_inputs + decoder_features
        init_state = self.get_init_state(batch_size)
        
        Warning("WARNING: confirm feature dim semantics make sense, " \
                "you dumb dumb!")

        if prefix_size > 0:
            prefix_inputs = []
            prev_outputs = []
            for input in decoder_inputs:
                if input.dim() == 1:
                    prefix_inputs.append(input)
                    prev_outputs.append(input)
                if input.dim() == 2:
                    prefix_inputs.append(input[:,:-1])
                    prev_outputs.append(input[:,-1])
                else:
                    raise Exception(
                        "I don't know what to do with " \
                        "input with dims = {}".format(input.dim()))
            prefix_logits, state = super(RNNLM, self).forward(
                prefix_inputs, prefix_inputs[0].size(1), prev_state=init_state)
        else: 
            state = init_state
            prev_outputs = []
            for input in decoder_inputs:
                if input.dim() == 1:
                    prev_outputs.append(input)
                if input.dim() == 2:
                    prev_outputs.append(input[:,-1])
                else:
                    raise Exception(
                        "I don't know what to do with " \
                        "input with dims = {}".format(input.dim()))

        # This getter is a BAD idea.
        Warning("Fix stop getter you dumb dumb.")
        stop_index = self.get_meta("input_reader").vocab.index("_STOP_")

        beam_prev_outputs = self.init_beam_outputs(prev_outputs, beam_size)
        beam_rnn_state = self.init_beam_rnn_state(state, beam_size)

        beam_scores = decoder_inputs[0].data.new(
            batch_size, beam_size).float()
        beam_scores.fill_(float("-inf"))            
        for batch in range(batch_size):
            beam_scores[batch, 0] = 0

        return self.beam_search(
            beam_prev_outputs, prev_state=beam_rnn_state, scores=beam_scores,
            max_steps=max_steps, beam_size=beam_size, batch_size=batch_size,
            stop_index=stop_index)
