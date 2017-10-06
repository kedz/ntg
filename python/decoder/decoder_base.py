from abc import ABC, abstractmethod
from torch import nn

class DecoderBase(nn.Module, ABC):
    def __init__(self):
        super(DecoderBase, self).__init__()

    


    #@abstractmethod
    #def greedy_predict(self, batch, return_logits=False, max_steps=100):
        #pass


