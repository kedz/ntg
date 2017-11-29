from dataset.dataset_base import Dataset
from collections import namedtuple
import torch
from torch.autograd import Variable


class MDSDataset(Dataset):
    def __init__(self, inputs, input_length, targets, texts, **args):
        super(MDSDataset, self).__init__(input_length, **args)
        
        self.texts_ = texts
        index = torch.LongTensor([i for i in range(inputs.size(0))])
        self.add_data(inputs, input_length, "inputs")
        self.add_data(targets, input_length, "targets")
        self.add_data(input_length, None, "input_length")
        self.add_data(index, None, "myindices")

        self.Batch_ = namedtuple(
            "Batch", 
            ["inputs", "input_length", "targets", "texts"])
        self.set_gpu(self.gpu)

    def set_batch_buffers(self):
        if self.gpu < 0:
            name2buffer = self.name2cpu_buffer_
        else:
            name2buffer = self.name2gpu_buffer_

        batch_texts = []
        #indices = name2buffer["myindices"]
        #for idx in indices:
        #    batch_texts.append(self.texts_[idx])

        inputs = Variable(name2buffer["inputs"])
        input_length = Variable(name2buffer["input_length"])
        targets = Variable(name2buffer["targets"]) 
        self.batch_ = self.Batch(inputs, input_length, targets, batch_texts)

    def get_batch(self):
        if self.gpu < 0:
            name2buffer = self.name2cpu_buffer_
        else:
            name2buffer = self.name2gpu_buffer_

        batch_texts = []
        indices = name2buffer["myindices"]
        del self.batch_.texts[:]
        for idx in indices:
            self.batch_.texts.append(self.texts_[idx])


        return self.batch_

    @property
    def Batch(self):
        return self.Batch_
