
from abc import ABC, abstractmethod

from models.model_base import ModelBase
from torch import nn

class SequenceModelBase(ModelBase):
    def __init__(self):
        super(SequenceModelBase, self).__init__()

    @abstractmethod
    def greedy_predict(self, batch, return_logits=False, max_steps=100):
        pass
    

