import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, input_size, filter_widths, num_filters, dropout=0.0, 
                 activation="relu"):
        super(EncoderCNN, self).__init__()

        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.filters_ = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (w, input_size)) 
             for w in filter_widths])
        
        if activation == "relu":
            self.activation_ = F.relu
        else:
            raise Exception(
                "activation {} not implemented for no good reason.".format(
                    activation))

        self.dropout_ = dropout
        self.output_size_ = num_filters * len(filter_widths)

    @property
    def output_size(self):
        return self.output_size_

    @property
    def filters(self):
        return self.filters_

    @property
    def activation(self):
        return self.activation_

    @property
    def dropout(self):
        return self.dropout_

    def encoder_state_output(self, inputs, length=None):
        batch_size = inputs.size(0)
        sequence_size = inputs.size(1)

        # Convolutional filters are batch_size x channels x height x width
        # and we only have 1 channel.
        inputs_4d = inputs.view(batch_size, 1, sequence_size, -1)

        feature_maps = []
        for filter in self.filters:
            
            feature_map = self.activation(filter(inputs_4d)).squeeze(3)
            map_size = feature_map.size(2)
            
            feature_map_pooled = F.max_pool1d(feature_map, map_size)
            feature_map_pooled.squeeze_(2)

            if self.dropout > 0:
                feature_map_pooled = F.dropout(
                    feature_map_pooled, training=self.training, p=self.dropout)
            
            feature_maps.append(feature_map_pooled)

        return torch.cat(feature_maps, 1)
