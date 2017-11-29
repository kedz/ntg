import torch
import torch.nn as nn
import torch.nn.functional as F
from layer_norm import LayerNorm

class CNNEncoder(nn.Module):

    def __init__(self, input_module, filter_widths, filter_outputs,
                 dropout=0.0, layer_norm=False,
                 activation="relu"):
        super(CNNEncoder, self).__init__()

        self.input_module_ = input_module
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.filters_ = nn.ModuleList(
            [nn.Conv2d(1, filter_outputs, (fw, input_module.embedding_size)) 
             for fw in filter_widths])
        
        if activation == "relu":
            self.activation_ = F.relu
            self.reverse_activation_and_pool_ = True
        else:
            raise Exception(
                "activation {} not implemented for no good reason.".format(
                    activation))

        self.dropout_ = dropout
        if layer_norm:
            self.layer_norms_ = nn.ModuleList([LayerNorm(filter_outputs)
                                               for _ in filter_widths])
        else:
            self.layer_norms_ = []

    @property
    def dropout(self):
        return self.dropout_

    @property
    def layer_norms(self):
        return self.layer_norms_

    def activation_and_pool(self, filter_output):
        max_steps = filter_output.size(2)
        if self.reverse_activation_and_pool:
            pool = F.max_pool1d(filter_output, max_steps).squeeze(2)
            output = self.activation(pool)
        else:
            conv_output = self.activation(filter_output)
            output = F.max_pool1d(conv_output, max_steps).squeeze(2)
           
        return output

    def forward(self, input):
        
        batch_size = input.size(0)
        max_steps = input.size(1)
        embedding_size = self.input_module.embedding_size

        input_sequence = self.input_module([input], transpose=False)
        conv_outputs = []
        for filter in self.filters:
            filter_output = filter(
                input_sequence.view(
                    batch_size, 1, max_steps, embedding_size)).squeeze(3)

            conv_out = self.activation_and_pool(filter_output)
            conv_outputs.append(conv_out)

        if len(self.layer_norms) > 0:
            conv_outputs = [ln(co) 
                            for ln, co in zip(self.layer_norms, conv_outputs)]

        if len(conv_outputs) > 1:
            output = torch.cat(conv_outputs, 1)
        else:
            output = conv_outputs[0]

        if self.dropout > 0:
            output = F.dropout(
                output, training=self.training, p=self.dropout, inplace=True)

        return output

    @property
    def activation(self):
        return self.activation_

    @property
    def reverse_activation_and_pool(self):
        return self.reverse_activation_and_pool_

    @property
    def input_module(self):
        return self.input_module_

    @property
    def filters(self):
        return self.filters_
