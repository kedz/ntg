from dataset.dataset_base import Dataset
from collections import namedtuple
import torch
from torch.autograd import Variable


class SimpleClassificationDataset(Dataset):
    def __init__(self, inputs, targets, **args):

        input_length = torch.LongTensor([1 for i in range(inputs.size(0))])
        super(SimpleClassificationDataset, self).__init__(input_length, **args)
        
        self.add_data(inputs, None, "inputs")
        self.add_data(targets, None, "targets")

        self.Batch_ = namedtuple(
            "Batch", 
            ["inputs", "targets"])
        self.set_gpu(self.gpu)

    def set_batch_buffers(self):
        if self.gpu < 0:
            name2buffer = self.name2cpu_buffer_
        else:
            name2buffer = self.name2gpu_buffer_

        inputs = Variable(name2buffer["inputs"])
        targets = Variable(name2buffer["targets"]) 
        self.batch_ = self.Batch(inputs, targets)

    def get_batch(self):
        if self.gpu < 0:
            name2buffer = self.name2cpu_buffer_
        else:
            name2buffer = self.name2gpu_buffer_

        return self.batch_

    @property
    def Batch(self):
        return self.Batch_
