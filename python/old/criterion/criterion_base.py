import torch
from abc import ABC, abstractmethod

class Criterion(ABC):
    def __init__(self, model, optimizer):
        super(Criterion, self).__init__()
        self.model_ = model
        self.optimizer_ = optimizer

    @property
    def optimizer(self):
        return self.optimizer_

    @property
    def model(self):
        return self.model_
    
    @abstractmethod
    def eval(self, logits, target):
        pass

    def minimize(self, batch):

        self.optimizer.zero_grad()
        
        # forward pass
        logits = self.model(batch.inputs)
        batch_loss = self.eval(logits, batch.targets)
        
        # backward pass
        batch_loss.backward()
        self.optimizer.step()
        
        return batch_loss.data[0]

    @abstractmethod
    def reset(self):
        pass

    @property
    def avg_loss(self):
        return float("nan")

    def status_msg(self):
        return "Average loss = {}".format(self.avg_loss)
