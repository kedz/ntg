from torch.autograd import Variable
from dataset.dataset_base import Dataset2
from collections import namedtuple

class Simple1DClassificationDataset(Dataset2):
    def __init__(self, inputs, targets, **args):
        super(Simple1DClassificationDataset, self).__init__(**args)
        self.register_data(inputs, None, "inputs")
        self.register_data(targets, None, "targets")

        self.Batch = namedtuple(
            "Batch", 
            ["inputs", "targets"])
        self.cpu_batch_ = self.Batch(
            Variable(self.cpu_buffers_[0]), 
            Variable(self.cpu_buffers_[1]))

    def get_batch(self):
        return self.cpu_batch_


#        self.set_gpu(self.gpu)


