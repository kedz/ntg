from sklearn.metrics import precision_recall_fscore_support


class MultiClassFMeasureReporter(object):
    def __init__(self, num_classes, labels=None, mode="logit",
                 report_all=True):
        if mode not in ["prob", "logit"]:
            raise Exception("mode must be either 'prob' or 'logit'.")

        self.report_all = report_all       
        self.num_classes_ = num_classes
        
        if labels is None:
            labels = ["class-{}".format(i) for i in range(1, num_classes + 1)]

        if len(labels) != num_classes:
            raise Exception("Number of classes must equal number of labels.")

        self.labels_ = tuple(labels)
        
        self.mode_ = mode
        self.name_ = "MultiClassFMeasureReporter"
        #self.total_examples_ = 0
        #self.total_correct_ = 0

        self.pred_output = []
        self.gold_output = []


    @property
    def num_classes(self):
        return self.num_classes_

    @property
    def labels(self):
        return self.labels_

    @property
    def mode(self):
        return self.mode_

    @property
    def name(self):
        return self.name_

    def update(self, output, expected):

        #if self.mode == "logit":

        #    prob = F.sigmoid(output.data)
        #else:
        #    prob = output.data

        _, pred_labels = output.data.max(1)

        self.pred_output.extend(pred_labels.tolist())
        self.gold_output.extend(expected.data.tolist())

    def reset(self):
        self.pred_output = []
        self.gold_output = []

    def result_dict(self):
        prfs_all = precision_recall_fscore_support(
            self.gold_output, self.pred_output, average=None,
            labels=[i for i in range(len(self.labels))])
 
        prfs_macro = precision_recall_fscore_support(
            self.gold_output, self.pred_output, average="macro",
            labels=[i for i in range(len(self.labels))])
        
        results = {self.name: {}}
        for l, label in enumerate(self.labels):

            prec = prfs_all[0][l]
            recall = prfs_all[1][l]
            fmeasure = prfs_all[2][l]
            results[self.name][label] = {"precision": prec, 
                                         "recall": recall, 
                                         "f-measure": fmeasure}

        results[self.name]["macro avg."] = {"precision": prfs_macro[0],
                                            "recall": prfs_macro[1],
                                            "f-measure": prfs_macro[2]}
        return results

    def report_string(self):
        rd = self.result_dict()

        
        label_len = max([len(label) for label in self.labels])
        tmp = "class: {label:" + str(label_len)  + "s} " \
            "prec: {precision:0.3f} recall: {recall:0.3f} " \
            "f-meas.: {f-measure:0.3f}"
        
        report_lines = []
        if self.report_all:
            for label in self.labels:
                report_lines.append(
                    tmp.format(**rd[self.name][label], label=label))
        report_lines.append(
            ("{label:"+ str(7 + label_len) +"s} " \
             "prec: {precision:0.3f} recall: {recall:0.3f} " \
             "f-meas.: {f-measure:0.3f}").format(
                **rd[self.name]["macro avg."], label="macro avg."))

        max_len = max([len(self.name)] + [len(line) for line in report_lines])
        
        for i in range(len(report_lines)):
            report_lines[i] = "| " + report_lines[i] \
                + " " * (max_len - len(report_lines[i])) + " |" 

        line1 = ".-" + self.name + "-" * (max_len - len(self.name)) + "-."
        line3 = ".-" + "-" * max_len + "-."
        lines = [line1] + report_lines + [line3]
        return lines, len(lines), len(line1)

    @property
    def initial_value(self):
        return 0.

    def is_better(self, new_value, old_value):
        return new_value > old_value

    def criterion_value_from_result_dict(self, result_dict):
        return result_dict[self.name]["macro avg."]["f-measure"]    

