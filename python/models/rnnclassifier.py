import torch
import torch.nn as nn
from torch.autograd import Variable

from models.model_base import ModelBase
from encoder import RNNEncoder
from mlp import MLP


class RNNClassifier(ModelBase):

    @classmethod
    def from_args(cls, args, encoder_input_modules,
                  dropout=None, rnn_type=None, target_vocab_size=None,
                  bidirectional=None, learn_init=None):

        if learn_init is None:
            learn_init = bool(args.learn_init)
        
        if target_vocab_size is None:
            target_vocab_size = args.target_vocab_size

        if dropout is None:
            dropout = args.dropout

        encoder = RNNEncoder.from_args(
            args, encoder_input_size=encoder_input_modules.embedding_size,
            dropout=dropout, rnn_type=rnn_type, bidirectional=bidirectional)

        mlp_input_size = 0
        for dim in encoder.rnn_state_dims:
            mlp_input_size += dim[0] * dim[2]

        mlp = MLP(mlp_input_size, target_vocab_size, dropout=dropout)

        return cls(encoder_input_modules, encoder, mlp, learn_init=learn_init)


    def __init__(self, encoder_input_modules, encoder, mlp, learn_init=False):

        super(RNNClassifier, self).__init__()

        self.encoder_input_modules_ = encoder_input_modules
        self.encoder_ = encoder
        self.mlp_ = mlp
        self.freeze_input_ = False
        self.freeze_encoder_ = False

        if learn_init:
            init_state = [nn.Parameter(torch.FloatTensor(*dims).fill_(0))
                          for dims in encoder.rnn_state_dims]
            self.init_state = nn.ParameterList(init_state)
        else:
            self.init_state = nn.ParameterList()


    def thawed_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("encoder_input_modules_.") \
                    and self.freeze_input_:
                continue
            if name.startswith("encoder_.") \
                    and self.freeze_encoder_:
                continue
            if name.startswith("init_state.") \
                    and self.freeze_encoder_:
                continue

            yield param

    def freeze_input_parameters(self, freeze=True):
        self.freeze_input_ = freeze

    def freeze_encoder_parameters(self, freeze=True):
        self.freeze_input_ = freeze

    @property
    def encoder_input_modules(self):
        return self.encoder_input_modules_

    @property
    def encoder(self):
        return self.encoder_

    @property
    def mlp(self):
        return self.mlp_

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

    def forward(self, batch):

        # todo add a size parameter to batch
        batch_size = batch.inputs[0].size(0)
        max_steps = batch.inputs[0].size(1)
        init_state = self.get_init_state(batch_size)

        encoder_inputs = self.encoder_input_modules.forward_sequence(
            batch.inputs, max_steps)

        _, encoder_state = self.encoder(
            encoder_inputs, batch.input_length, prev_state=init_state)

        if isinstance(encoder_state, (tuple, list)):
            flat_states = [s.permute(1, 0, 2).contiguous().view(batch_size, -1)
                           for s in encoder_state]
            mlp_input = torch.cat(flat_states, 1)
        else:
            mlp_input = encoder_state.permute(1, 0, 2).contiguous().view(
                batch_size, -1)

        logits = self.mlp(mlp_input)
        
        return logits
