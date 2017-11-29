from abc import ABC
from torch import nn

class ModelBase(nn.Module, ABC):
    def __init__(self):
        super(ModelBase, self).__init__()
        self.model_meta_ = {}

    def set_meta(self, key, value):
        self.model_meta_[key] = value

    def get_meta(self, key):
        return self.model_meta_[key]
          
    @property
    def meta(self):
        return self.model_meta_
