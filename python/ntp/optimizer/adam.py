import torch.optim
from .optimizer_base import OptimizerBase

class Adam(OptimizerBase):
    def __init__(self, parameters, lr=.001, weight_decay=0.0):
        super(Adam, self).__init__()
        self.parameters_ = parameters
        self.lr_ = lr
        self.weight_decay_ = weight_decay 
        self.optimizer_ = torch.optim.Adam(
            parameters, lr=lr, weight_decay=self.weight_decay)

    @property
    def weight_decay(self):
        return self.weight_decay_

    @property
    def lr(self):
        return self.lr_

    @property
    def parameters(self):
        return self.parameters_

    @property
    def optimizer(self):
        return self.optimizer_

    def reset(self):
        self.optimizer_ = torch.optim.Adam(
            self.parameters, lr=self.lr, weight_decay=self.weight_decay)

    def __getattr__(self, attr):
        return getattr(self.optimizer, attr)
