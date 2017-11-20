import torch
import torch.nn.functional as F

# TODO make owkr with masks
# TODO add tests
class BinaryAccuracyReporter(object):

    def __init__(self, mode="prob"):
        Warning("BinaryAccuracyReporter does not handle masked data yet.")

        if mode not in ["prob", "logit"]:
            raise Exception("mode must be either 'prob' or 'logit'.")

        self.mode_ = mode
        self.name_ = "BinaryAccuracyReport"
        self.total_examples_ = 0
        self.total_correct_ = 0


    @property
    def mode(self):
        return self.mode_

    @property
    def name(self):
        return self.name_

    def update(self, output, expected):
        if self.mode == "logit":
            prob = F.sigmoid(output.data)
        else:
            prob = output.data

        pred_labels = prob.gt(.5).type_as(expected.data)
        num_correct = pred_labels.data.eq(expected.data).sum()
        self.total_correct_ += num_correct
        self.total_examples_ += expected.data.numel()

    def reset(self):
        self.total_correct_ = 0
        self.total_examples_ = 0

    def result_dict(self):
        if self.total_examples_ == 0:
            acc = 0
        else:
            acc = self.total_correct_ / self.total_examples_
        return {self.name: {"accuracy": acc}}
