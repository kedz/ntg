from dataset.dataset_base import Dataset
from collections import namedtuple
from torch.autograd import Variable


class SequenceClassificationDataset(Dataset):
    def __init__(self, inputs, input_length, targets, **args):

        super(SequenceClassificationDataset, self).__init__(
            input_length, **args)

        self.input_names = []
        self.target_names = []

        for i, input in enumerate(inputs, 1):
            name = "input_{}".format(i)
            self.input_names.append(name)
            self.add_data(
                input, input_length, name)
        self.add_data(input_length, None, "input_length")

        for i, target in enumerate(targets, 1):
            name = "target_{}".format(i)
            self.target_names.append(name)
            self.add_data(
                target, None, "target_{}".format(i))

        self.Batch_ = namedtuple(
            "Batch", 
            ["inputs", "input_length", "targets"])
        self.set_gpu(self.gpu)

    def set_batch_buffers(self):
        if self.gpu < 0:
            name2buffer = self.name2cpu_buffer_
        else:
            name2buffer = self.name2gpu_buffer_

        inputs = [Variable(name2buffer[name]) 
                  for name in self.input_names]
        input_length = Variable(name2buffer["input_length"])

        targets = [Variable(name2buffer[name]) 
                   for name in self.target_names]

        self.batch_ = self.Batch(inputs, input_length, targets)

    def get_batch(self):
        return self.batch_

    @property
    def Batch(self):
        return self.Batch_
