import torch
import torch.nn as nn
import torch.nn.functional as F

def identity(input):
    return input

class FeedForwardLayer(nn.Module):
    def __init__(self, input_size, output_size, activation="relu",
                 dropout=0.0):
        super(FeedForwardLayer, self).__init__()
        if activation is None:
            self.activation_ = identity
        elif activation == "relu":
            self.activation_ = F.relu
        elif activation == "tanh":
            self.activation_ = F.tanh
        elif activation == "sigmoid":
            self.activation_ = F.sigmoid
        else:
            raise Exception("activation {} not recognized.".format(activation))

        self.input_size_ = input_size 
        self.output_size_ = output_size
        self.linear_module_ = nn.Linear(input_size, output_size)
        self.dropout_ = dropout 

    def forward(self, inputs):
        preactivation = self.linear_module_(inputs)
        output = self.activation_(preactivation)
        
        if self.dropout > 0:
            output = F.dropout(output, p=self.dropout, training=self.training)

        return output

    @property
    def dropout(self):
        return self.dropout_

    @property
    def input_size(self):
        return self.input_size_

    @property
    def output_size(self):
        return self.output_size_



