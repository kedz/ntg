from abc import ABC, abstractmethod

from models.model_base import ModelBase
from torch import nn

class ClassificationModel(ModelBase):
    def __init__(self, encoder, mlp):
        super(ClassificationModel, self).__init__()

        self.encoder_ = encoder
        self.mlp_ = mlp

    def forward(self, batch):

        encoder_output = self.encoder(batch.input)
        logits = self.mlp(encoder_output)

        return logits

    def predict(self, input):
        pass
    
    @property
    def encoder(self):
        return self.encoder_

    @property
    def mlp(self):
        return self.mlp_
