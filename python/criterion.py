import torch
from abc import ABC, abstractmethod
import torch.nn.functional as F

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
        logits = self.model(batch)
        batch_loss = self.eval(logits, batch.target)
        
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




class CrossEntropy(Criterion):

    def __init__(self, model, optimizer, weight=None):
        super(CrossEntropy, self).__init__(model, optimizer)

        self.weight = weight
        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0

    def reset(self):
        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0

    def eval(self, logits, target):
        batch_loss = F.cross_entropy(
            logits, target, weight=self.weight, ignore_index=-1, 
            size_average=True)

        batch_size = logits.size(0)
        self.tot_cross_entropy_ += batch_loss.data[0] * batch_size
        self.tot_examples_ += batch_size

        return batch_loss

    def compute_loss(self, batch):
        logits = self.model(batch)
        batch_loss = self.eval(logits, batch.target)
        return batch_loss

    @property
    def avg_loss(self):
        if self.tot_examples_ > 0:
            return self.tot_cross_entropy_ / self.tot_examples_
        else:
            return float("nan")
 

class PrecRecallReporter(Criterion):
    def __init__(self, criterion, class_names):
        super(PrecRecallReporter, self).__init__(
            criterion.model, criterion.optimizer)
        self.criterion_ = criterion
        self.class_names_ = class_names
        self.true_positives_ = torch.LongTensor(len(class_names)).fill_(0)
        self.false_positives_ = torch.LongTensor(len(class_names)).fill_(0)

   
    @property
    def criterion(self):
        return self.criterion_

    def eval(self, logits, target):
        batch_loss = self.criterion.eval(logits, target)
        print(logits.max(1))
        
        exit()

    def compute_loss(self, batch):
        logits = self.model(batch)
        batch_loss = self.criterion.eval(logits, batch)
        return batch_loss
        
    @property
    def avg_loss(self):
        return self.criterion.avg_loss


    def reset(self):
        self.true_positives_.fill_(0)
        self.false_positives_.fill_(0)
        self.criterion.reset()

class SequenceCrossEntropy(object):

    def __init__(self, model, optimizer):

        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0
        self.model_ = model
        self.optimizer_ = optimizer

    def reset(self):
        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0

    def compute_loss(self, logits, targets):

        max_steps = logits.size(0)
        batch_size = logits.size(1)
        vocab_size = logits.size(2)
        total_loss = 0

        logits_flat = logits.view(max_steps * batch_size, vocab_size)
        targets_flat = targets.contiguous().view(max_steps * batch_size)
        total_loss = F.cross_entropy(
            logits_flat, targets_flat, ignore_index=0, size_average=False)

        total_elements = targets.data.gt(0).sum()
        batch_loss = total_loss / total_elements
 
        self.tot_examples_ += total_elements
        self.tot_cross_entropy_ += total_loss.data[0]

        return batch_loss

    @property
    def optimizer(self):
        return self.optimizer_

    @property
    def model(self):
        return self.model_

    def minimize(self, batch):

        self.optimizer.zero_grad()
        logits = self.model(batch)
        batch_loss = self.compute_loss(logits, batch.target.t())
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss.data[0]

    def eval(self, batch):
        logits = self.model(batch)
        batch_loss = self.compute_loss(logits, batch.target.t())
        return batch_loss.data[0]

    @property
    def avg_loss(self):
        if self.tot_examples_ > 0:
            return self.tot_cross_entropy_ / self.tot_examples_
        else:
            return float("nan")
