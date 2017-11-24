from .data_layout import DataLayout

import torch
from torch.autograd import Variable

import random
from collections import namedtuple


class Dataset(object):
    def __init__(self, *tensors, layout=None, metadata=None, batch_size=1, 
                 shuffle=True, gpu=-1):
        
        self.data_ = []
        self.length_ = []
        self.name_ = []

        self.metadata_ = metadata

        self.cpu_buffers_ = []
        self.name2cpu_buffer_ = {}
        self.name2gpu_buffer_ = {}
        
        self.gpu_ = gpu
        self.shuffle_ = shuffle
        self.batch_size_ = batch_size

        for tensor_data in tensors:
            if len(tensor_data) == 2:
                self.register_data(tensor_data[0], None, tensor_data[1])
            elif len(tensor_data) == 3:
                self.register_data(*tensor_data)
            else:
                raise Exception(
                    "tensors must be iterable of data, size, name.")

        if layout == None:
            layout = [[name, name] for name in self.name_]

        name2data = {name: data for name, data in zip(self.name_, self.data_)}
        self.data_layout_ = DataLayout(layout, name2data, "Dataset")
        self.cpu_batch_ = DataLayout(layout, self.name2cpu_buffer_, "CPUBatch")
        
    def index_select(self, index):
        if isinstance(index, (list, tuple)):
            index = torch.LongTensor(index)
        layout = self.data_layout_.layout_meta
        data = [(t.index_select(0, index), l, n) 
                for t, l, n in zip(self.data_, self.length_, self.name_)]
        gpu = self.gpu
        batch_size = self.batch_size
        shuffle = self.shuffle
        metadata = None
        if self.metadata is not None:
            metadata = [self.metadata[idx] for idx in index]

        return Dataset(*data, layout=layout, batch_size=batch_size,
                       metadata=metadata, shuffle=shuffle, gpu=gpu)

    def register_data(self, tensor, length, name):
        self.data_.append(tensor)
        self.length_.append(length)
        self.name_.append(name) 
        self.cpu_buffers_.append(Variable(tensor.new()))
        self.name2cpu_buffer_[name] = self.cpu_buffers_[-1]

    def iter_batch(self):

        indices = [i for i in range(self.size)]
        if self.shuffle:
            random.shuffle(indices)
        indices = torch.LongTensor(indices)

        for p in range(0, self.size, self.batch_size):
            indices_batch = indices[p:p + self.batch_size]


            for j in range(len(self.data_)):
                length = self.length_[j]
                if length is not None:
                    max_len = length.index_select(
                        0, indices_batch, out=self.max_len_buf_).max()
                    buffer = self.data_[j][:,:max_len].index_select(
                        0, indices_batch, out=self.cpu_buffers_[j].data)
                else:
                    buffer = self.data_[j].index_select(
                        0, indices_batch, out=self.cpu_buffers_[j].data)

                if self.gpu_ > -1:
                    self.gpu_buffers_[j].resize_(buffer.size()).copy_(buffer)
            # TODO make this lazy somehow
            if self.metadata is not None:
                md_batch = [self.metadata[idx] for idx in indices_batch]
                self.cpu_batch_.metadata = md_batch

            yield self.cpu_batch_
 
    @property
    def metadata(self):
        return self.metadata_
   
    @property
    def shuffle(self):
        return self.shuffle_ 

    @shuffle.setter
    def shuffle(self, val):
        self.shuffle_ = bool(val)

    @property
    def gpu(self):
        return self.gpu_

    @property
    def batch_size(self):
        return self.batch_size_
    
    @batch_size.setter
    def batch_size(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise Exception("batch_size must be a positive integer.")
        self.batch_size_ = batch_size
        
    @property
    def size(self):
        return self.data_[0].size(0)
 
    def __getattr__(self, attr):
        return getattr(self.data_layout_, attr)
