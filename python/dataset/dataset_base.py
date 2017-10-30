import torch
from torch.autograd import Variable
import random
from collections import namedtuple
from abc import ABC, abstractmethod


class Dataset(ABC):

    def __init__(self, sorting_length, batch_size=1, chunk_size=500, gpu=-1):
        super(ABC, self).__init__()

        self.sorting_length_ = sorting_length

        self.batch_size_ = batch_size
        self.chunk_size_ = chunk_size

        self.data_ = []
        self.length_ = []
        self.name_ = []
        
        self.cpu_buffers_ = []
        self.name2cpu_buffer_ = {}
        self.name2gpu_buffer_ = {}
        self.max_len_buf_ = torch.LongTensor()
        self.gpu_ = gpu

    @abstractmethod
    def set_batch_buffers(self):
        pass

    @abstractmethod
    def get_batch(self):
        pass

    @property
    def sorting_length(self):
        return self.sorting_length_

    def set_gpu(self, gpu):
        self.gpu_ = gpu
        if self.gpu_ > -1:
            self.gpu_buffers_ = []
            for cpu_buffer, name in zip(self.cpu_buffers, self.name_):
                self.gpu_buffers_.append(cpu_buffer.new().cuda(self.gpu_))
                self.name2gpu_buffer_[name] = self.gpu_buffers_[-1]
        self.set_batch_buffers()

    def set_batch_size(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise Exception("Batch size must be a positive integer.")
        self.batch_size_ = batch_size
    
    @property
    def gpu(self):
        return self.gpu_

    @property
    def batch_size(self):
        return self.batch_size_

    @property
    def chunk_size(self):
        return self.chunk_size_

    @property
    def size(self):
        return self.sorting_length.size(0)
    
    @property
    def cpu_buffers(self):
        return self.cpu_buffers_

    def add_data(self, tensor, length, name):
        self.data_.append(tensor)
        self.length_.append(length)
        self.name_.append(name) 
        self.cpu_buffers_.append(tensor.new())
        self.name2cpu_buffer_[name] = self.cpu_buffers_[-1]

    def iter_batch(self):

        indices = [i for i in range(self.size)]
        random.shuffle(indices)
        indices = torch.LongTensor(indices)
        
        for i in range(0, self.size, self.batch_size * self.chunk_size):
            indices_chunk = indices[i:i + self.batch_size * self.chunk_size]
            lengths_batch = self.sorting_length.index_select(0, indices_chunk)
            sorted, sort_indices = torch.sort(
                lengths_batch, dim=0, descending=True)
            indices_sorted_ = indices_chunk.index_select(0, sort_indices)
            indices[i:i + self.batch_size * self.chunk_size] = indices_sorted_

        for p in range(0, self.size, self.batch_size):
            indices_batch = indices[p:p + self.batch_size]

            for j in range(len(self.data_)):
                length = self.length_[j]
                if length is not None:
                    max_len = length.index_select(
                        0, indices_batch, out=self.max_len_buf_).max()
                    buffer = self.data_[j][:,:max_len].index_select(
                        0, indices_batch, out=self.cpu_buffers_[j])
                else:
                    buffer = self.data_[j].index_select(
                        0, indices_batch, out=self.cpu_buffers_[j])

                if self.gpu_ > -1:
                    self.gpu_buffers_[j].resize_(buffer.size()).copy_(buffer)

            yield self.get_batch()
                


