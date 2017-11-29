from criterion.criterion_base import Criterion
import torch.nn.functional as F

class SequenceCrossEntropy(Criterion):

    def __init__(self, model, optimizer, weight=None):
        super(SequenceCrossEntropy, self).__init__(model, optimizer)

        self.weight_ = weight
        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0
        self.tot_sequences_ = 0

    @property
    def weight(self):
        return self.weight_

    def reset(self):
        self.tot_cross_entropy_ = 0
        self.tot_examples_ = 0
        self.tot_sequences_ = 0

    def eval(self, logits, targets):

        if isinstance(targets,  (list, tuple)):
            targets = targets[0]

        max_steps = logits.size(0)
        batch_size = logits.size(1)
        vocab_size = logits.size(2)

        logits_flat = logits.view(max_steps * batch_size, vocab_size)
        targets_tr_flat = targets.t().contiguous().view(max_steps * batch_size)
        total_loss = F.cross_entropy(
            logits_flat, targets_tr_flat, ignore_index=0, size_average=False)

        total_elements = targets.data.gt(0).sum()
        batch_loss = total_loss / total_elements
 
        self.tot_examples_ += total_elements
        self.tot_cross_entropy_ += total_loss.data[0]
        self.tot_sequences_ += batch_size

        return batch_loss

    def compute_loss(self, batch):
        logits = self.model(batch)
        batch_loss = self.eval(logits, batch.targets)
        return batch_loss

    @property
    def avg_loss(self):
        if self.tot_examples_ > 0:
            return self.tot_cross_entropy_ / self.tot_examples_
        else:
            return float("nan")

    @property
    def avg_sequence_loss(self):
        if self.tot_sequences_ > 0:
            return self.tot_cross_entropy_ / self.tot_sequences_
        else:
            return float("nan")

    def status_msg(self):
        return "NLL/step = {:0.4f}\nNLL/seq  = {:0.4f}".format(
            self.avg_loss, self.avg_sequence_loss)

    def result_dict(self):
        return {"nll": self.avg_loss, "nll/seq": self.avg_sequence_loss}
