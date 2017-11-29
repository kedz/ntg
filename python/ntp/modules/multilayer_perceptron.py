from .feedforward_layer import FeedForwardLayer

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, output_size, 
                 output_activation="sigmoid",
                 hidden_sizes=None, hidden_layer_activations="relu",
                 hidden_layer_dropout=0.0, output_layer_dropout=0.0):

        super(MultiLayerPerceptron, self).__init__()

        self.input_size_ = input_size
        self.output_size_ = output_size
        self.output_activation_ = output_activation
        self.hidden_layer_dropout_ = hidden_layer_dropout
        self.output_layer_dropout_ = output_layer_dropout

        if hidden_sizes is None:
            hidden_sizes = ()  
        elif isinstance(hidden_sizes, int):
            hidden_sizes = tuple([hidden_sizes])
        elif isinstance(hidden_sizes, list):
            hidden_sizes = tuple(hidden_sizes)
        elif not isinstance(hidden_sizes, tuple):
            raise Exception(
                "hidden_sizes must be int or a list/tuple of ints.")
        self.hidden_sizes_ = hidden_sizes

        if isinstance(hidden_layer_activations, (list, tuple)):
            if not len(hidden_layer_activations) == len(self.hidden_sizes):
                raise Exception("hidden_layer_activations must be same size " \
                                "as hidden_sizes")
            hidden_layer_activations = tuple(hidden_layer_activations)

        elif isinstance(hidden_layer_activations, (type(None), str)):
            hidden_layer_activations = tuple(
                [hidden_layer_activations] * len(hidden_sizes))

        else:
            raise Exception("hidden_layer_activations must be None, string, " \
                            "or list/tuple of None and strings.")

        self.hidden_layer_activations_ = hidden_layer_activations

        current_size = self.input_size

        hidden_layers = []
        for next_size, act in zip(self.hidden_sizes, 
                                  self.hidden_layer_activations):
            hidden_layers.append(
                FeedForwardLayer(
                    current_size, next_size, activation=act, 
                    dropout=self.hidden_layer_dropout))
            current_size = next_size

        self.hidden_layers_ = nn.ModuleList(hidden_layers)

        self.output_layer_ = FeedForwardLayer(
            current_size, self.output_size, activation=self.output_activation,
            dropout=self.output_layer_dropout)

    def forward(self, inputs):

        next_inputs = inputs

        for hidden_layer in self.hidden_layers:
            next_inputs = hidden_layer(next_inputs)

        output = self.output_layer(next_inputs)

        return output

    @property
    def input_size(self):
        return self.input_size_

    @property
    def output_size(self):
        return self.output_size_

    @property
    def output_activation(self):
        return self.output_activation_

    @property
    def hidden_sizes(self):
        return self.hidden_sizes_

    @property
    def hidden_layer_activations(self):
        return self.hidden_layer_activations_

    @property
    def hidden_layers(self):
        return self.hidden_layers_

    @property
    def output_layer(self):
        return self.output_layer_

    @property
    def hidden_layer_dropout(self):
        return self.hidden_layer_dropout_

    @property
    def output_layer_dropout(self):
        return self.output_layer_dropout_
