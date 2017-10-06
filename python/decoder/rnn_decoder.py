import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder.decoder_base import DecoderBase
import attention


class RNNDecoder(DecoderBase):

    @classmethod
    def from_args(cls, args, decoder_input_size=None, dropout=None, 
                  attention_type=None, rnn_type=None, rnn_hidden_size=None,
                  num_layers=None, target_vocab_size=None):

        if num_layers is None:
            num_layers = args.num_layers

        if decoder_input_size is None:
            decoder_input_size = args.decoder_input_size

        if rnn_hidden_size is None:
            rnn_hidden_size = args.rnn_hidden_size

        if dropout is None:
            dropout = args.dropout

        if attention_type is None:
            attention_type = args.attention_type

        if rnn_type is None:
            rnn_type = args.rnn_type

        if target_vocab_size is None:
            target_vocab_size = args.target_vocab_size

        rnn_args = {"num_layers": num_layers, 
                    "bidirectional": False,
                    "dropout": dropout}
        
        if rnn_type == "rnn":
            rnn_init = nn.RNN
            rnn_args["nonlinearity"] = "relu"
            rnn_state_dims = ((num_layers, 1, rnn_hidden_size),)
        elif rnn_type == "gru":
            rnn_init = nn.GRU
            rnn_state_dims = ((num_layers, 1, rnn_hidden_size),)
        elif rnn_type == "lstm":
            rnn_init = nn.LSTM 
            rnn_state_dims = ((num_layers, 1, rnn_hidden_size),) * 2
        else:
            raise Exception("rnn_type {} not supported".format(rnn_type))

        rnn_module = rnn_init(
            decoder_input_size, rnn_hidden_size, **rnn_args)
        
        mlp_input_size = rnn_hidden_size
        
        if attention_type == "none":
            attention_module = attention.NoOpAttention()
        elif attention_type == "dot":
            attention_module = attention.DotAttention(merge_type="concat")
            mlp_input_size *= 2
        else:
            raise Exception(
                "attention_type {} not implemented.".format(
                    args.attention_type))

        predictor_module = nn.Linear(mlp_input_size, target_vocab_size)

        return cls(rnn_module, attention_module, predictor_module,
            rnn_state_dims=rnn_state_dims)       

    def __init__(self, rnn_module, attention_module, predictor_module, 
                 rnn_state_dims=None):
        super(RNNDecoder, self).__init__()

        self.rnn_module_ = rnn_module
        self.attention_module_ = attention_module
        self.predictor_module_ = predictor_module
        
        if rnn_state_dims is None:
            rnn_state_dims = ()    
        self.rnn_state_dims_ = rnn_state_dims
        
    @property
    def rnn_state_dims(self):
        return self.rnn_state_dims_

    @property
    def rnn_module(self):
        return self.rnn_module_

    @property
    def attention_module(self):
        return self.attention_module_

    @property
    def predictor_module(self):
        return self.predictor_module_

    def forward(self, inputs, prev_state=None, context=None): 

        # test for dims = 3

        max_steps = inputs.size(0)
        batch_size = inputs.size(1)

        rnn_output, rnn_state = self.rnn_module(inputs, prev_state)

        attention_output = self.attention_module(
            rnn_output, context=context, return_weights=False)

        attention_output_flat = attention_output.view(
            max_steps * batch_size, attention_output.size(2))

        logits_flat = self.predictor_module(attention_output_flat)
        logits = logits_flat.view(max_steps, batch_size, logits_flat.size(1))
        return logits

#    def greedy_predict(self, init_state, context):
#        pass
#
#    def forward_step(self, inputs, prev_state=None, context=None):
#
#        input_sequence = self.input_module(inputs)
#        rnn_output, rnn_state = self.rnn_module(input_sequence, prev_state)
#        attention_output, weights = self.attention_module(
#            rnn_output, context=context)
#        logits = self.predictor_module(attention_output[0])
#        return logits, weights, rnn_state



