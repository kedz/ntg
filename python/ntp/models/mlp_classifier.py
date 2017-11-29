from ..modules import MultiLayerPerceptron
import torch.nn.functional as F


class MLPClassifier(MultiLayerPerceptron):
    def __init__(self, input_size_or_module, num_classes, mode="multiclass",
                 hidden_sizes=None, hidden_layer_activations="relu",
                 hidden_layer_dropout=0.0):

        if isinstance(input_size_or_module, int):
            self.input_size_ = input_size_or_module
            self.input_module_ = None
        else:
            raise Exception()

        self.mode = mode

        super(MLPClassifier, self).__init__(
            self.input_size, num_classes, 
            hidden_sizes=hidden_sizes,
            hidden_layer_dropout=hidden_layer_dropout,
            hidden_layer_activations=hidden_layer_activations,
            output_activation=None)

    def predict(self, inputs):
        logits = self.forward(inputs)
        if self.mode == "multiclass":
            _, predicted = logits.max(1)
            return predicted

        else:
            predicted = F.sigmoid(logits).gt(.5).long()
            return predicted

    def predict_prob(self, inputs):
        logits = self.forward(inputs)
        if self.mode == "multiclass":
            return F.softmax(logits)
        else:
            return F.sigmoid(logits)
