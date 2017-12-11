import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_cell="lstm", dropout=0.0,
                 bidirectional=False, layers=1, rnn_activation="relu",
                 merge_mode="concat"):

        super(EncoderRNN, self).__init__()

        if rnn_cell == "rnn":
            self.rnn_ = nn.RNN(
                input_size, hidden_size, dropout=dropout, 
                bidirectional=bidirectional, num_layers=layers,
                activation=rnn_activation)
        elif rnn_cell == "lstm":
            self.rnn_ = nn.LSTM(
                input_size, hidden_size, dropout=dropout, 
                bidirectional=bidirectional, num_layers=layers)
        elif rnn_cell == "gru":
            self.rnn_ = nn.GRU(
                input_size, hidden_size, dropout=dropout, 
                bidirectional=bidirectional, num_layers=layers)
        else:
            raise Exception(
                "rnn_cell does not support value {}".format(rnn_cell))

        self.merge_mode = merge_mode

    @property
    def merge_mode(self):
        return self.merge_mode_

    @merge_mode.setter
    def merge_mode(self, mode):
        if mode not in ["concat", "add", "mean"]:
            raise Exception(
                "merge_mode must be one of 'concat', 'add', or 'mean'")
        self.merge_mode_ = mode

    @property
    def rnn(self):
        return self.rnn_

    def forward(self, inputs, length=None, prev_state=None,
                return_state=True, return_context=True):
        
        context, state = self.apply_encoder(
            inputs, length=length, prev_state=prev_state)
        if return_state and not return_context:
            return state
        elif not return_state and return_context:
            return context
        elif return_state and return_context:
            return context, state
        else:
            raise Exception("Can't return nothing!")

    def apply_encoder(self, inputs, length=None, prev_state=None):
        if length is None:
            raise Exception("Implement me!")
        else:
            inputs_packed = nn.utils.rnn.pack_padded_sequence(
                inputs, length.data.tolist(), batch_first=False)
            output_packed, state = self.rnn(inputs_packed, prev_state)
            return output_packed, state 

    def encoder_state(self, inputs, length=None, prev_state=None):
        return self(inputs, length=length, prev_state=prev_state, 
                    return_context=False) 

    def encoder_state_output(self, inputs, length=None, prev_state=None):
        state = self.encoder_state(
            inputs, length=length, prev_state=prev_state)
        output = state[0]

        if self.rnn.bidirectional:
            output_fwd = output[-2]
            output_bwd = output[-1]
            if self.merge_mode == "concat": 
                output_cat = torch.cat([output_fwd, output_bwd], 1)
                return output_cat
            elif self.merge_mode == "add":
                return output_fwd + output_bwd
            elif self.merge_mode == "mean":
                output_cat = torch.cat([output_fwd, output_bwd], 1)
                return output_cat.view(output_fwd.size(0), 2, -1).mean(1)
        else:
            return output[-1]

    @property
    def output_size(self):
        if self.rnn.bidirectional and self.merge_mode == "concat":
            return self.rnn.hidden_size * 2
        else:
            return self.rnn.hidden_size
