from .crf_criterion_base import CRFCriterionBase
import torch
import torch.nn.functional as F


class CrossEntropy(CRFCriterionBase):
    def __init__(self, weight=None, mask_value=-1,
                 name="SequenceMultiClassCrossEntropy"):
        super(CrossEntropy, self).__init__(name)
        self.weight = weight
        self.mask_value = mask_value

        self.tot_examples_ = 0
        self.tot_cross_entropy_ = 0

    def eval(self, model, batch):

        gold_states = batch.targets.t().contiguous()
        scores = model.score_state(batch.inputs, gold_states, normalized=True)
        total_elements = batch.inputs.length.data.sum()
        xent = -scores.sum()
        batch_loss = xent / total_elements

        self.tot_examples_ += total_elements
        self.tot_cross_entropy_ += xent.data[0]

        pred_states = model.predict(batch.inputs)

        return batch_loss, pred_states

    def reset_statistics(self):

        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0

    @property
    def initial_value(self):
        return float("inf")

    def is_better(self, new_value, old_value):
        if new_value < old_value:
            return True
        else:
            return False

    @property
    def avg_loss(self):
        if self.tot_examples_ > 0:
            return self.tot_cross_entropy_ / self.tot_examples_
        else:
            return float("nan")

    def criterion_value_from_result_dict(self, result_dict):
        return result_dict[self.name]["criterion"]    
