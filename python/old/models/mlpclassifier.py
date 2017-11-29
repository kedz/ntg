from models.model_base import ModelBase
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(ModelBase):
    '''
    MLP classifier. If an output size is 1, will treat the
    output type as a logistic sigmoid unit. Otherwise, softmax units
    will be used for the output.

    '''
    def __init__(self, input_size, output_size, 
                 hidden_sizes=None, activation_functions=None,
                 input_dropout=0.0, dropout=0.0):
        
        super(MLPClassifier, self).__init__()

        self.input_size_ = input_size
        self.output_size_ = output_size
        
        self.input_layer_ = nn.Linear(input_size, output_size)

    def forward(self, inputs):

        output = logits = self.input_layer_(inputs)
        if self.output_size == 1:
            output = F.sigmoid(logits)

        return output

    @property
    def input_size(self):
        return self.input_size_

    @property
    def output_size(self):
        return self.output_size_
