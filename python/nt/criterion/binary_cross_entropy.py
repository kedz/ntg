from nt.criterion import CriterionBase
import torch
import torch.nn.functional as F


# TODO add tests for exception throwing.
class BinaryCrossEntropy(CriterionBase):

    def __init__(self, mode="prob", weight=None, mask_value=None):
        super(BinaryCrossEntropy, self).__init__()
        
        if mode not in ["prob", "logit"]:
            raise Exception("Invalid mode")
        
        if weight is not None:
            if weight.size(0) != 2 or weight.dim() != 1:
                raise Exception("weight must be a 1-d tensor of size 2.")

        if mask_value in [0, 1]:
            raise Exception("mask_value must be an integer besides 0 and 1.")
        if mask_value is not None and not isinstance(mask_value, int):
            raise Exception("mask_value must be an integer besides 0 and 1.")

        self.mask_value_ = mask_value
        self.mode_ = mode
        self.weight_ = weight
        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0

      

    @property
    def mask_value(self):
        return self.mask_value_

    @mask_value.setter
    def mask_value(self, mask_value):
        if mask_value in [0, 1]:
            raise Exception("mask_value must be an integer besides 0 and 1.")
        if mask_value is not None and not isinstance(mask_value, int):
            raise Exception("mask_value must be an integer besides 0 and 1.")
        self.mask_value_ = mask_value

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

        total_elements = targets.data.numel()
        output_weight = None
        
        if self.mask_value is not None:
            output_weight = targets.data.new().resize_(targets.data.size())
            output_weight.fill_(1)
            mask = targets.data.eq(self.mask_value)
            output_weight.masked_fill_(mask, 0)
            total_elements -= mask.sum()

        if self.weight is not None:
            if output_weight is None:
                output_weight = self.weight.new().resize_(targets.size())
                output_weight.fill_(0)

            pos_mask = torch.eq(targets.data, 1)
            neg_mask = torch.eq(targets.data, 0)
            output_weight.masked_fill_(pos_mask, self.weight[1])
            output_weight.masked_fill_(neg_mask, self.weight[0])

        if self.mode == "prob":
            total_loss = F.binary_cross_entropy(
                output, targets, weight=output_weight, 
                size_average=False)
        else:

            if output_weight is not None:
                output_weight = torch.autograd.Variable(output_weight)
            total_loss = F.binary_cross_entropy_with_logits(
                output, targets, weight=output_weight,
                size_average=False)

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


   
