import torch.nn.functional as F



# TODO make work with masks
# TODO add tests
# TODO don't use scikit learn, store the necessary stats more efficiently

from sklearn.metrics import precision_recall_fscore_support
#(y_true, y_pred, beta=1.0, labels=None, pos_label=1, average=None, warn_for=(‘precision’, ’recall’, ’f-score’), sample_weight=None)
class BinaryFMeasureReporter(object):

    def __init__(self, mode="prob"):
        Warning("BinaryFMeasureReporter does not handle masked data yet.")

        if mode not in ["prob", "logit"]:
            raise Exception("mode must be either 'prob' or 'logit'.")

        self.mode_ = mode
        self.name_ = "BinaryFMeasureReporter"
        #self.total_examples_ = 0
        #self.total_correct_ = 0

        self.pred_output = []
        self.gold_output = []

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

        pred_labels = prob.gt(.5).data.tolist()
        self.pred_output.extend(pred_labels)
        self.gold_output.extend(expected.data.tolist())

    def reset(self):
        self.pred_output = []
        self.gold_output = []

    def result_dict(self):
        prfs = precision_recall_fscore_support(
            self.gold_output, self.pred_output, average="binary")
        prec = prfs[0]
        recall = prfs[1]
        fmeasure = prfs[2]

        return {self.name: {"precision": prec, 
                            "recall": recall, 
                            "f-measure": fmeasure}} 

    def report_string(self):
        rd = self.result_dict()

        tmp = "prec: {precision:0.3f} recall: {recall:0.3f} " \
            "f-meas.: {f-measure:0.3f}"
        
        line2 = tmp.format(**rd[self.name])
        max_len = max(len(self.name), len(line2))
        line1 = ".-" + self.name + "-" * (max_len - len(self.name)) + "-."
        line2 = "| " + line2 + " |"
        line3 = ".-" + "-" * max_len + "-."
        lines = [line1, line2, line3]
        return lines, 3, len(line1)

