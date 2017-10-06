from dataset.dataset_base import Dataset
from collections import namedtuple
from torch.autograd import Variable

class SequencePredictionDataset(Dataset):
    def __init__(self, decoder_inputs, targets, target_length,
                 encoder_inputs=None, encoder_length=None, 
                 decoder_features=None, **args):

        super(SequencePredictionDataset, self).__init__(
            target_length, **args)

        print("Warning not implemented for seq2seq yet.")

        self.decoder_input_names = []
        self.target_names = []

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

        self.Batch_ = namedtuple("Batch", ["decoder_inputs", "targets"])
        self.set_gpu(self.gpu)

    def set_batch_buffers(self):
        if self.gpu < 0:
            name2buffer = self.name2cpu_buffer_
        else:
            name2buffer = self.name2gpu_buffer_

        decoder_inputs = [Variable(name2buffer[name]) 
                          for name in self.decoder_input_names]
        targets = [Variable(name2buffer[name]) 
                   for name in self.target_names]

        self.batch_ = self.Batch(decoder_inputs, targets)

    def get_batch(self):
        return self.batch_

    @property
    def Batch(self):
        return self.Batch_




