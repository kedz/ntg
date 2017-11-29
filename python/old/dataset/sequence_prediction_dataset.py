from dataset.dataset_base import Dataset
from collections import namedtuple
from torch.autograd import Variable


class SequencePredictionDataset(Dataset):
    def __init__(self, decoder_inputs, targets, target_length,
                 encoder_inputs=None, encoder_length=None, 
                 decoder_features=None, **args):

        if encoder_length is not None:
            length = encoder_length
        else:
            length = target_length

        super(SequencePredictionDataset, self).__init__(length, **args)

        self.encoder_input_names = []
        self.decoder_feature_names = []
        self.decoder_input_names = []
        self.target_names = []

        if len(encoder_inputs) > 0:
            if encoder_length is None:
                raise Exception("Must provide encoder input length.")
            for i, encoder_input in enumerate(encoder_inputs, 1):
                name = "encoder_input_{}".format(i)
                self.encoder_input_names.append(name)
                self.add_data(
                    encoder_input, encoder_length, name)
            self.add_data(
                encoder_length, None, "encoder_length")

        for i, decoder_input in enumerate(decoder_inputs, 1):
            name = "decoder_input_{}".format(i)
            self.decoder_input_names.append(name)
            self.add_data(
                decoder_input, target_length, name)

        for i, target in enumerate(targets, 1):
            name = "target_{}".format(i)
            self.target_names.append(name)
            self.add_data(
                target, target_length, "target_{}".format(i))

        if decoder_features is not None:
            for i, decoder_feature in enumerate(decoder_features, 1):
                name = "decoder_feature_{}".format(i)
                self.decoder_feature_names.append(name)
                self.add_data(
                    decoder_feature, None, name)

        self.Batch_ = namedtuple(
            "Batch", 
            ["encoder_inputs", "encoder_length", "decoder_inputs", "targets", 
             "decoder_features"])
        self.set_gpu(self.gpu)

    def set_batch_buffers(self):
        if self.gpu < 0:
            name2buffer = self.name2cpu_buffer_
        else:
            name2buffer = self.name2gpu_buffer_

        encoder_inputs = [Variable(name2buffer[name]) 
                          for name in self.encoder_input_names]
        if len(encoder_inputs) > 0:
            encoder_length = Variable(name2buffer["encoder_length"])
        else:
            encoder_length = None

        decoder_inputs = [Variable(name2buffer[name]) 
                          for name in self.decoder_input_names]
        targets = [Variable(name2buffer[name]) 
                   for name in self.target_names]
        decoder_features = [Variable(name2buffer[name]) 
                            for name in self.decoder_feature_names]

        self.batch_ = self.Batch(
            encoder_inputs, encoder_length, decoder_inputs, 
            targets, decoder_features)

    def get_batch(self):
        return self.batch_

    @property
    def Batch(self):
        return self.Batch_
