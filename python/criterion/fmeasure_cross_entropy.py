from criterion.criterion_base import Criterion
from criterion.cross_entropy import CrossEntropy
import torch.nn.functional as F
import torch

from sklearn.metrics import precision_recall_fscore_support

class FMeasureCrossEntropy(Criterion):

    def __init__(self, model, optimizer, num_classes, weight=None, 
                 class_names=None):
        super(FMeasureCrossEntropy, self).__init__(model, optimizer)
        self.cross_entropy_ = CrossEntropy(model, optimizer, weight=weight)
        self.tp_ = torch.FloatTensor(num_classes).zero_()
        self.fp_ = torch.FloatTensor(num_classes).zero_()
        self.fn_ = torch.FloatTensor(num_classes).zero_()
        
        if class_names is None:
            class_names = ["class_{}".format(i) for i in range(num_classes)]
        self.class_names = class_names

        self.preds = []
        self.golds = []

    @property
    def cross_entropy(self):
        return self.cross_entropy_

    @property
    def weight(self):
        return self.cross_entropy.weight

    def reset(self):
        self.cross_entropy.reset()
        self.preds = []
        self.golds = []

    def eval(self, logits, targets):

        if isinstance(targets,  (list, tuple)):
            targets = targets[0]
       
        prediction = logits.max(1)[1]
        
        self.preds.extend(prediction.data.tolist())
        self.golds.extend(targets.data.tolist())

        return self.cross_entropy.eval(logits, targets)

    def compute_loss(self, batch):
        logits = self.model(batch)
        batch_loss = self.eval(logits, batch.targets)
        return batch_loss

    @property
    def avg_loss(self):
        prfs = precision_recall_fscore_support(self.golds, self.preds)
        macro_f1 = prfs[2].mean()
        return macro_f1

    def status_msg(self):
        prfs = precision_recall_fscore_support(self.golds, self.preds)

        msg_lines = []
        for i, name in enumerate(self.class_names):
            msg_lines.append("{:15s} P:{:0.3f} R:{:0.3f} F:{:0.3f}".format(
                name, prfs[0][i], prfs[1][i], prfs[2][i]))
        msg_lines.append("NLL = {:0.4f}".format(self.cross_entropy.avg_loss))

        return "\n".join(msg_lines)

    def result_dict(self):
        prfs = precision_recall_fscore_support(self.golds, self.preds)
        result = {}
        for i, name in enumerate(self.class_names):
            result[name] = {"precision": prfs[0][i], 
                            "recall": prfs[1][i], 
                            "f-measure": prfs[2][i]}
        result["macro"] = {"precision": prfs[0].mean(), 
                           "recall": prfs[1].mean(), 
                           "f-measure": prfs[2].mean()}
        result["nll"] = self.cross_entropy.avg_loss
        return result
