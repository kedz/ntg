import torch.nn as nn
import torch.nn.functional as F


def from_args(args, rnn_hidden_size=None, num_layers=None, bidirectional=None,
              bridge_type=None):

    if bidirectional is None:
        bidirectional = args.bidirectional

    if rnn_hidden_size is None:
        rnn_hidden_size = args.rnn_hidden_size

    if num_layers is None:
        num_layers = args.num_layers

    if bridge_type is None:
        bridge_type = args.bridge_type
        
    if bidirectional:
        if bridge_type.startswith("linear-"):
            act_type = bridge_type.split("-")[1]
            return FullyConnectedBridge(
                rnn_hidden_size, num_layers, 
                activation=act_type, bidirectional_input=True)
        elif bridge_type == "average":
            return AveragingBridge(num_layers)
        else:
            raise Exception(
                "bridge_type {} not valid.".format(bridge_type))
    else:
        return NoOpBridge()

class FullyConnectedBridge(nn.Module):

    def __init__(self, state_size, num_layers, bidirectional_input=True, 
                 activation="relu"):
        super(FullyConnectedBridge, self).__init__()

        in_size = state_size * num_layers * (2 if bidirectional_input else 1)
        out_size = state_size * num_layers 
        self.state_size_ = state_size
        self.num_layers_ = num_layers

        self.fc_layer_ = nn.Linear(in_size, out_size)
        if activation == "relu":
            self.activation_ = F.relu
        elif activation == "tanh":
            self.activation_ = F.tanh
        else:
            raise Exception("activation {} not supported".format(activation))

    @property
    def num_layers(self):
        return self.num_layers_

    @property
    def state_size(self):
        return self.state_size_

    def transform_state(self, state):
        batch_size = state.size(1)
        state_flat = state.permute(1, 0, 2).contiguous().view(batch_size, -1)
        output_flat = self.activation_(self.fc_layer_(state_flat))
        output = output_flat.view(
            batch_size, self.state_size, self.num_layers)
        return output.permute(2,0,1).contiguous()

    def forward(self, input):
        return self.transform_state(input)

class AveragingBridge(nn.Module):
    def __init__(self, num_layers):
        super(AveragingBridge, self).__init__()
        self.num_layers_ = num_layers

    @property
    def num_layers(self):
        return self.num_layers_
     
    def transform_state(self, state):
        batch_size = state.size(1)
        state_size = state.size(2)
        state4d = state.view(self.num_layers, 2, batch_size, state_size)
        output = state4d.mean(1)
        return output

    def forward(self, input):
        # lstm state is a tuple of tensors.
        if isinstance(input, (tuple, list)):
            return tuple([self.transform_state(state) for state in input])
        else:
            return self.transform_state(input)

class NoOpBridge(nn.Module):
    def forward(self, input):
        return input
