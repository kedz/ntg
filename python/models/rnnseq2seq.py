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

       
