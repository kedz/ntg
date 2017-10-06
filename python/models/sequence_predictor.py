from abc import abstractmethod
from models.model_base import ModelBase

class SequencePredictor(ModelBase):
    def __init__(self, input_modules, decoder):
        super(SequencePredictor, self).__init__()
        self.input_modules_ = input_modules
        self.decoder_ = decoder

    @property
    def input_modules(self):
        return self.input_modules_

    @property
    def decoder(self):
        return self.decoder_

    def forward(self, inputs, prev_state=None, context=None):
        decoder_inputs = self.input_modules(inputs)
        logits = self.decoder(
            decoder_inputs, prev_state=prev_state, context=context)
        return logits

    def greedy_predict(self):
        pass

    def beam_search(self):
        pass
