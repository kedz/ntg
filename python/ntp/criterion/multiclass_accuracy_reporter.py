import torch
import torch.nn.functional as F

# TODO make owkr with masks
# TODO add tests
class MultiClassAccuracyReporter(object):

    def __init__(self, mask_value=-1):
        Warning("Doesn't do masking of sequences.")

        self.name_ = "MultiClassAccuracyReporter"
        self.total_examples_ = 0
        self.total_correct_ = 0
        self.mask_value = mask_value

    @property
    def name(self):
        return self.name_

    def update(self, output, expected):
        _, pred_label = output.max(1)
        num_correct = pred_label.eq(expected).data.sum()
        total_examples = expected.data.numel()
        self.total_correct_ += num_correct
        self.total_examples_ += total_examples

    def reset(self):
        self.total_correct_ = 0
        self.total_examples_ = 0

    def result_dict(self):
        if self.total_examples_ == 0:
            acc = 0
        else:
            acc = self.total_correct_ / self.total_examples_
        return {self.name: {"accuracy": acc}}

    def report_string(self):
        rd = self.result_dict()

        tmp = "Accuracy: {accuracy:0.3f}"
        
        line2 = tmp.format(**rd[self.name])
        max_len = max(len(self.name), len(line2))
        line1 = ".-" + self.name + "-" * (max_len - len(self.name)) + "-."
        line2 = "| " + line2 + " " * (max_len - len(line2))  + " |"
        line3 = ".-" + "-" * max_len + "-."
        lines = [line1, line2, line3]
        return lines, 3, len(line1)

