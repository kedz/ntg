from .multiclass_cross_entropy import MultiClassCrossEntropy
import torch
import torch.nn.functional as F


class SequenceMultiClassCrossEntropy(MultiClassCrossEntropy):
    def __init__(self, mode="logit", weight=None, mask_value=-1,
                 name="SequenceMultiClassCrossEntropy"):
        super(SequenceMultiClassCrossEntropy, self).__init__(
            mode=mode, weight=weight, mask_value=mask_value, name=name)

    def eval(self, output, targets):

        total_elements = targets.data.ne(self.mask_value).sum()

        batch_size = output.size(0)
        sequence_size = output.size(1)
        total_size = batch_size * sequence_size
        output_flat = output.view(total_size, -1)
        targets_flat = targets.view(total_size)
        
        if self.mode == "prob":
            total_loss = F.nll_loss(
                torch.log(output_flat), targets_flat, weight=self.weight,
                size_average=False, ignore_index=self.mask_value)
        else:
            total_loss = F.cross_entropy(
                output_flat, targets_flat, weight=self.weight,
                size_average=False, ignore_index=self.mask_value)

        batch_loss = total_loss / total_elements
        self.tot_examples_ += total_elements
        self.tot_cross_entropy_ += total_loss.data[0]

        return batch_loss
