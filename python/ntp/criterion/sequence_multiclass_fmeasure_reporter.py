from .multiclass_fmeasure_reporter import MultiClassFMeasureReporter

class SequenceMultiClassFMeasureReporter(MultiClassFMeasureReporter):
    def __init__(self, num_classes, labels=None, mode="logit", mask_value=-1,
                 report_all=True):
        super(SequenceMultiClassFMeasureReporter, self).__init__(
            num_classes, labels=labels, mode=mode, report_all=report_all)

        self.mask_value_ = mask_value

    @property
    def mask_value(self):
        return self.mask_value_

    def update(self, output, expected):
        mask = expected.ne(self.mask_value)
        _, pred_labels = output.max(2)

        self.pred_output.extend(
            pred_labels.masked_select(mask).data.tolist())
        self.gold_output.extend(expected.masked_select(mask).data.tolist())
