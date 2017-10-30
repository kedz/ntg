import torch
import torch.nn as nn
from torch.autograd import Variable

from models.sequence_predictor import SequencePredictor
from encoder import RNNEncoder
from decoder import RNNDecoder
import bridge as be 
from parallel_module import ParallelModule

class RNNSeq2Seq(SequencePredictor):

    @classmethod
    def from_args(cls, args, encoder_input_modules, decoder_input_modules,
                  dropout=None, rnn_type=None, target_vocab_size=None,
                  attention_type=None, bidirectional=None, learn_init=None,
                  bridge_type=None):

        if learn_init is None:
            learn_init = bool(args.learn_init)

        if bridge_type is None:
            bridge_type = args.bridge_type

        encoder = RNNEncoder.from_args(
            args, encoder_input_size=encoder_input_modules.embedding_size,
            dropout=dropout, rnn_type=rnn_type, bidirectional=bidirectional)

        if args.rnn_type == "lstm":
            bridge1 = be.from_args(
                args, bridge_type=bridge_type, 
                bidirectional=bidirectional)
            bridge2 = be.from_args(
                args, bridge_type=bridge_type, 
                bidirectional=bidirectional)
            bridge = ParallelModule([bridge1, bridge2])

        else:
            bridge = be.from_args(
                args, bridge_type=bridge_type, 
                bidirectional=bidirectional)

        decoder = RNNDecoder.from_args(
            args, decoder_input_size=decoder_input_modules.embedding_size,
            dropout=dropout, rnn_type=rnn_type,
            target_vocab_size=target_vocab_size,
            attention_type=attention_type)
        return cls(encoder_input_modules, decoder_input_modules,
                   encoder, bridge, decoder, learn_init=learn_init)

    def __init__(self, encoder_input_modules, decoder_input_modules, 
                 encoder, bridge, decoder, learn_init=False):

        super(RNNSeq2Seq, self).__init__(decoder_input_modules, decoder)

        self.encoder_input_modules_ = encoder_input_modules
        self.encoder_ = encoder
        self.bridge_ = bridge

        if learn_init:
            init_state = [nn.Parameter(torch.FloatTensor(*dims).fill_(0))
                          for dims in encoder.rnn_state_dims]
            self.init_state = nn.ParameterList(init_state)
        else:
            self.init_state = nn.ParameterList()

    @property
    def encoder_input_modules(self):
        return self.encoder_input_modules_

    @property
    def encoder(self):
        return self.encoder_

    @property
    def bridge(self):
        return self.bridge_

    # TODO make this a mixin, also for rnnlm model
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

        # todo add a size parameter to batch
        batch_size = batch.encoder_inputs[0].size(0)
        encoder_max_steps = batch.encoder_inputs[0].size(1)
        decoder_max_steps = batch.decoder_inputs[0].size(1)
        
        init_state = self.get_init_state(batch_size)

        encoder_inputs = self.encoder_input_modules.forward_sequence(
            batch.encoder_inputs, encoder_max_steps)

        context, encoder_state = self.encoder(
            encoder_inputs, batch.encoder_length, prev_state=init_state)

        decoder_state = self.bridge(encoder_state)

        logits, _ = super(RNNSeq2Seq, self).forward(
            batch.decoder_inputs + batch.decoder_features,
            decoder_max_steps, 
            prev_state=decoder_state,
            context=context)

        if mask_output:
            mask = batch.decoder_inputs[0].data.t().eq(0)
            mask3d = mask.view(decoder_max_steps, batch_size, 1).expand(
                decoder_max_steps, batch_size, logits.data.size(2))
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

    def init_beam_context(self, context, beam_size):
        steps = context.size(0)
        batch_size = context.size(1)
        hidden_size = context.size(2)
        beam_context = context.repeat(1, 1, beam_size).view(
            steps, batch_size * beam_size, hidden_size)
        return beam_context


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



    def complete_sequence(self, encoder_inputs, encoder_length, 
                          decoder_inputs, decoder_features, 
                          max_steps=100, beam_size=8):
         
        batch_size = encoder_inputs[0].size(0)
        encoder_max_steps = encoder_inputs[0].size(1)
        prefix_size = decoder_inputs[0].size(1) - 1  

        init_state = self.get_init_state(batch_size)

        encoder_inputs = self.encoder_input_modules.forward_sequence(
            encoder_inputs, encoder_max_steps)

        context, encoder_state = self.encoder(
            encoder_inputs, encoder_length, prev_state=init_state)

        
        decoder_state = self.bridge(encoder_state)
        
        if prefix_size > 0:
            prefix_inputs = []
            prev_outputs = []
            for input in decoder_inputs + decoder_features:
                if input.dim() == 1:
                    prefix_inputs.append(input)
                    prev_outputs.append(input)
                elif input.dim() == 2:
                    prefix_inputs.append(input[:,:-1])
                    prev_outputs.append(input[:,-1])
                else:
                    raise Exception(
                        "I don't know what to do with " \
                        "input with dims = {}".format(input.dim()))
            prefix_logits, decoder_state = super(RNNSeq2Seq, self).forward(
                prefix_inputs, prefix_inputs[0].size(1), 
                prev_state=decoder_state,
                context=context)

        else: 
            prev_outputs = []
            for input in decoder_inputs + decoder_features:
                if input.dim() == 1:
                    prev_outputs.append(input)
                elif input.dim() == 2:
                    prev_outputs.append(input[:,-1])
                else:
                    raise Exception(
                        "I don't know what to do with " \
                        "input with dims = {}".format(input.dim()))


        # This getter is a BAD idea.
        Warning("Fix stop getter you dumb dumb.")
        stop_index = self.get_meta("target_reader").vocab.index("_DSTOP_")

        beam_prev_outputs = self.init_beam_outputs(prev_outputs, beam_size)
        beam_rnn_state = self.init_beam_rnn_state(decoder_state, beam_size)
        beam_context = self.init_beam_context(context, beam_size)

        beam_scores = decoder_inputs[0].data.new(
            batch_size, beam_size).float()
        beam_scores.fill_(float("-inf"))            
        for batch in range(batch_size):
            beam_scores[batch, 0] = 0

        return self.beam_search(
            beam_prev_outputs, prev_state=beam_rnn_state, scores=beam_scores,
            max_steps=max_steps, beam_size=beam_size, batch_size=batch_size,
            stop_index=stop_index, context=beam_context)
