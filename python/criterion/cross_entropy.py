from criterion.criterion_base import Criterion
import torch.nn.functional as F

class CrossEntropy(Criterion):

    def __init__(self, model, optimizer, weight=None):
        super(CrossEntropy, self).__init__(model, optimizer)

        self.weight_ = weight
        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0

    @property
    def weight(self):
        return self.weight_

    def reset(self):
        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0

    def eval(self, logits, targets):

        if isinstance(targets,  (list, tuple)):
            targets = targets[0]

        total_loss = F.cross_entropy(
            logits, targets, ignore_index=-1, weight=self.weight, 
            size_average=False)

        total_elements = targets.size(0)
        batch_loss = total_loss / total_elements
        self.tot_examples_ += total_elements
        self.tot_cross_entropy_ += total_loss.data[0]

        return batch_loss

    def compute_loss(self, batch):
        logits = self.model(batch.inputs)
        batch_loss = self.eval(logits, batch.targets)
        return batch_loss

    @property
    def avg_loss(self):
        if self.tot_examples_ > 0:
            return self.tot_cross_entropy_ / self.tot_examples_
        else:
            return float("nan")

    def status_msg(self):
        return "NLL = {:0.4f}".format(self.avg_loss)
