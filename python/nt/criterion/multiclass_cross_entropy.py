from .criterion_base import CriterionBase
import torch
import torch.nn.functional as F


# TODO add tests for exception throwing.
class MultiClassCrossEntropy(CriterionBase):

    def __init__(self, mode="logit", weight=None, mask_value=-1):
        super(MultiClassCrossEntropy, self).__init__()

        if mode not in ["prob", "logit"]:
            raise Exception("Invalid mode")

        self.mode_ = mode
        self.weight_ = weight
        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0
        self.mask_value_ = mask_value

    @property
    def mask_value(self):
        return self.mask_value_

    @property
    def mode(self):
        return self.mode_

    @property
    def weight(self):
        return self.weight_

    def reset_statistics(self):

        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0

    def eval(self, output, targets):

        total_elements = targets.data.ne(self.mask_value).sum()

        if self.mode == "prob":
            total_loss = F.nll_loss(
                torch.log(output), targets, weight=self.weight,
                size_average=False, ignore_index=self.mask_value)
        else:

            total_loss = F.cross_entropy(
                output, targets, weight=self.weight,
                size_average=False, ignore_index=self.mask_value)

        batch_loss = total_loss / total_elements
        self.tot_examples_ += total_elements
        self.tot_cross_entropy_ += total_loss.data[0]

        return batch_loss

    @property
    def avg_loss(self):
        if self.tot_examples_ > 0:
            return self.tot_cross_entropy_ / self.tot_examples_
        else:
            return float("nan")
