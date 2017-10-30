
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0,
                 layer_sizes=None, layer_activations=None):
        super(MLP, self).__init__()

        self.dropout_ = dropout
        self.input_size_ = input_size
        self.output_size_ = output_size

        linear_layers = []
        activations = []

        if layer_sizes is None:
            layer_sizes = []
        if layer_activations is None:
            layer_activations = []

        if len(layer_sizes) != len(layer_activations):
            raise Exception(
                "Must have same number of layer sizes and activations")

        for output_size, act in zip(layer_sizes, layer_activations):
            linear_layers.append(nn.Linear(input_size, output_size))
            
            if act == "relu":
                activations.append(F.relu)
            else:
                raise Exception("Activation {} not implemented.".format(act))
            
            input_size = output_size

        self.inner_layers_ = nn.ModuleList(linear_layers)
        self.inner_activations_ = activations
        self.final_layer_ = nn.Linear(input_size, self.output_size)

    @property
    def dropout(self):
        return self.dropout_

    def forward(self, input):
        hidden = input

        if self.dropout > 0:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        for linear, act in zip(self.inner_layers, self.inner_activations_):
            hidden = act(linear(hidden))
            if self.dropout > 0:
                hidden = F.dropout(
                    hidden, p=self.dropout, training=self.training, 
                    inplace=True)

        output = self.final_layer(hidden)
        return output

    @property
    def input_size(self):
        return self.input_size_

    @property
    def output_size(self):
        return self.output_size_

    @property
    def inner_layers(self):
        return self.inner_layers_

    @property
    def inner_activations(self):
        return selinf.inner_activations_

    @property
    def final_layer(self):
        return self.final_layer_
