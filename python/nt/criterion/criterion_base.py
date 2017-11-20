import torch
from abc import ABC, abstractmethod

class CriterionBase(ABC):
    def __init__(self, name="CriterionBase"):
        super(CriterionBase, self).__init__()
        self.reporters_ = []
        self.name_ = name

    @property
    def name(self):
        return self.name_

    @abstractmethod
    def eval(self, pred_output, target):
        pass

    def add_reporter(self, reporter):
        self.reporters_.append(reporter)

    def minimize(self, batch, model, opt):

        opt.zero_grad()
        
        # forward pass
        pred_output = model(batch.inputs)

        for reporter in self.reporters_:
            reporter.update(pred_output, batch.targets)

        batch_loss = self.eval(pred_output, batch.targets)
        
        # backward pass
        batch_loss.backward()

        opt.step()
        
        return batch_loss.data[0]

    def compute_loss(self, batch, model):
        pred_output = model(batch.inputs)
        for reporter in self.reporters_:
            reporter.update(pred_output, batch.targets)
        batch_loss = self.eval(pred_output, batch.targets)
        return batch_loss

    @abstractmethod
    def reset_statistics(self):
        pass

    def reset(self):
        for reporter in self.reporters_:
            reporter.reset()
        self.reset_statistics()

    def result_dict(self):
        result = {self.name: {"criterion": self.avg_loss}}
        for reporter in self.reporters_:
            result.update(reporter.result_dict())
        return result

    @property
    def avg_loss(self):
        return float("nan")
