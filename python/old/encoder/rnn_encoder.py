import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNEncoder(nn.Module):

    @classmethod
    def from_args(cls, args, encoder_input_size=None, dropout=None,
                  rnn_type=None, rnn_hidden_size=None, num_layers=None,
                  bidirectional=None):

        if num_layers is None:
            num_layers = args.num_layers

        if encoder_input_size is None:
            encoder_input_size = args.encoder_input_size

        if rnn_hidden_size is None:
            rnn_hidden_size = args.rnn_hidden_size

        if dropout is None:
            dropout = args.dropout

        if rnn_type is None:
            rnn_type = args.rnn_type

        if bidirectional is None:
            bidirectional = args.bidirectional

        rnn_args = {"num_layers": num_layers, 
                    "bidirectional": bidirectional,
                    "dropout": dropout}
        
        layers_x_dir = num_layers * 2 if bidirectional else num_layers

        if rnn_type == "rnn":
            rnn_init = nn.RNN
            rnn_args["nonlinearity"] = "relu"
            rnn_state_dims = ((layers_x_dir, 1, rnn_hidden_size),)
        elif rnn_type == "gru":
            rnn_init = nn.GRU
            rnn_state_dims = ((layers_x_dir, 1, rnn_hidden_size),)
        elif rnn_type == "lstm":
            rnn_init = nn.LSTM 
            rnn_state_dims = ((layers_x_dir, 1, rnn_hidden_size),) * 2
        else:
            raise Exception("rnn_type {} not supported".format(rnn_type))

        rnn_module = rnn_init(
            encoder_input_size, rnn_hidden_size, **rnn_args)
        return cls(rnn_module, bidirectional_merge_mode="average",
                   rnn_state_dims=rnn_state_dims)


    def __init__(self, rnn_module, bidirectional_merge_mode=None, 
                 rnn_state_dims=None):

        super(RNNEncoder, self).__init__()

        self.rnn_module_ = rnn_module
        
        self.bidirectional_merge_mode_ = bidirectional_merge_mode
        if bidirectional_merge_mode not in ["average", None]:
            raise Exception(
                "bidirectional_merge_mode {} not supported.".format(
                    bidirectional_merge_mode))
        
        if rnn_state_dims is None:
            rnn_state_dims = ()
        self.rnn_state_dims_ = rnn_state_dims

    @property
    def bidirectional_merge_mode(self):
        return self.bidirectional_merge_mode_

    @property
    def rnn_state_dims(self):
        return self.rnn_state_dims_

    @property
    def rnn_module(self):
        return self.rnn_module_

    def set_dropout(self, dropout):
        self.rnn_module.dropout = dropout


    def average_bidirectional_output(self, output):
        max_steps = output.size(0)
        batch_size = output.size(1)
        hidden_size = output.size(2) // 2
        output4d = output.view(max_steps, batch_size, 2, hidden_size)
        output_mean = output4d.mean(2)
        return output_mean

    def forward(self, input, input_length, prev_state=None):

        input_packed = nn.utils.rnn.pack_padded_sequence(
            input, input_length.data.tolist(), batch_first=False)

        output_packed, state = self.rnn_module(input_packed, prev_state)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output_packed, batch_first=False)
        
        if self.bidirectional_merge_mode == "average":
            output = self.average_bidirectional_output(output)

        return output, state
