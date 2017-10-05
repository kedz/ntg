import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNEncoder(nn.Module):

    def __init__(self, input_module, rnn_type, hidden_size, 
                 num_layers, bidirectional=True):

        super(RNNEncoder, self).__init__()

        self.input_module_ = input_module
        self.rnn_type_ = rnn_type
        self.hidden_size_ = hidden_size
        self.num_layers_ = num_layers
        self.bidirectional_ = bidirectional

        self.reset_module()

    def reset_module(self):

        rnn_args = {"num_layers": self.num_layers, 
                    "bidirectional": self.bidirectional}    
        if self.rnn_type == "rnn":
            rnn_init = nn.RNN
            rnn_args["nonlinearity"] = "relu"
        elif self.rnn_type == "gru":
            rnn_init = nn.GRU
        elif self.rnn_type == "lstm":
            rnn_init = nn.LSTM 
        else:
            raise Exception("rnn_type {} not supported".format(rnn_type))

        self.rnn_ = rnn_init(self.embedding_size, self.hidden_size, **rnn_args)

    @property
    def input_module(self):
        return self.input_module_

    @property
    def vocab_size(self):
        return self.input_module.vocab_size

    @property
    def embedding_size(self):
        return self.input_module.embedding_size

    @property
    def rnn_type(self):
        return self.rnn_type_

    @property
    def hidden_size(self):
        return self.hidden_size_

    @property
    def num_layers(self):
        return self.num_layers_

    @property
    def bidirectional(self):
        return self.bidirectional_

    def average_bidirectional_output(self, output):
        max_steps = output.size(0)
        batch_size = output.size(1)
        output4d = output.view(max_steps, batch_size, 2, self.hidden_size)
        output_mean = output4d.mean(2)
        return output_mean

    def forward(self, inputs, input_length, init_state=None):
        
        input_sequence = self.input_module(inputs, input_length=input_length)
        output_packed, state = self.rnn_(input_sequence, init_state)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output_packed, batch_first=False)

        if self.bidirectional:
            output = self.average_bidirectional_output(output)

        return output, state
