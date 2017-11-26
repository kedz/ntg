import torch
from abc import ABC, abstractmethod
from collections import defaultdict

class CriterionBase(ABC):
    def __init__(self, name="CriterionBase"):
        super(CriterionBase, self).__init__()
        self.reporters_ = []
        self.name_ = name
        self.checkpoints_ = defaultdict(list)

    def checkpoint(self, checkpoint_label):
        rd = self.result_dict()
        self.checkpoints_[checkpoint_label].append(rd)

    def best_checkpoint(self, checkpoint_label):
        min_obj = float('inf')
        min_checkpoint = -1
        for i, rd in enumerate(self.checkpoints_[checkpoint_label], 1):
            obj = rd[self.name]["criterion"] 
            if obj < min_obj:
                min_obj = obj
                min_checkpoint = i
        return min_checkpoint, min_obj

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

    def report(self, indent=""):
        blocks = []
        blocks.append({"lines": defaultdict(list), "width": 0, })
        
        current_block = 0
        max_width = 80

        for reporter in [self] + self.reporters_ :
            if hasattr(reporter, "report_string"):
                lines, row, cols = reporter.report_string()

                while True:
                    
                    if cols > max_width:
                        current_block = len(blocks)
                        blocks.append({"lines": defaultdict(list), "width": 0})
                        Warning("Overlength reporter: {}!".format(
                            reporter.name))
                        break
                    
                    block_width = blocks[current_block]["width"]
                    if block_width + cols > max_width:
                        current_block += 1
                        if len(blocks) == current_block:
                            blocks.append(
                                {"lines": defaultdict(list), "width": 0})
                    else:
                        break

                blocks[current_block]["width"] += cols + 2
                for i, line in enumerate(lines):
                    blocks[current_block]["lines"][i].append(line)
                    blocks[current_block]["lines"][i].append("  ")

        output_lines = []
        for block in blocks:
            for i in range(len(block["lines"])):
                output_lines.append(indent + "".join(block["lines"][i][:-1]))
        return "\n".join(output_lines)

    @property
    def avg_loss(self):
        return float("nan")

    def report_string(self):

        tmp = "Criterion: {:0.6f}"
        
        line2 = tmp.format(self.avg_loss)
        max_len = max(len(self.name), len(line2))
        line1 = ".-" + self.name + "-" * (max_len - len(self.name)) + "-."
        line2 = "| " + line2 + " " * (max_len - len(line2))  + " |"
        line3 = ".-" + "-" * max_len + "-."
        lines = [line1, line2, line3]
        return lines, 3, len(line1)
