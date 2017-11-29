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
                

#class SequencePredictionDataset(Dataset):
#    def __init__(self, decoder_inputs, decoder_targets, 
#                 encoder_inputs=None, decoder_features=None):



class SequenceClassificationDataset(Dataset):
    def __init__(self, input, input_length, target, **args):
        super(SequenceClassificationDataset, self).__init__(
            input_length, **args)
        self.add_data(input, input_length, "input")
        self.add_data(input_length, None, "input_length")
        self.add_data(target, None, "target")
        
        self.Batch_ = namedtuple(
            "Batch", ["input", "input_length", "target"])
        self.set_gpu(self.gpu)

    def set_batch_buffers(self):
        if self.gpu < 0:
            name2buffer = self.name2cpu_buffer_
        else:
            name2buffer = self.name2gpu_buffer_

        self.batch_ = self.Batch(
            Variable(name2buffer["input"]), 
            Variable(name2buffer["input_length"]), 
            Variable(name2buffer["target"]))
            
    def get_batch(self):
        return self.batch

    @property
    def Batch(self):
        return self.Batch_

    @property
    def batch(self):
        return self.batch_





class Seq2SeqDataset(Dataset):

    def __init__(self, encoder_inputs, encoder_input_length, decoder_inputs,
                 target, target_length, **keys):

        super(Seq2SeqDataset, self).__init__(encoder_input_length, **keys)

        for tensor, name in encoder_inputs:
            self.add_data(tensor, encoder_input_length, name)
        self.add_data(encoder_input_length, None, "encoder_input_length")
        for tensor, name in decoder_inputs:
            self.add_data(tensor, target_length, name)
        self.add_data(target, target_length, "target")

        self.encoder_inputs_ = encoder_inputs
        self.encoder_input_length_ = encoder_input_length
        self.decoder_inputs_ = decoder_inputs
        self.target_ = target
        self.target_length_ = target_length

        self.Batch_ = namedtuple(
            "Batch", ["encoder_inputs", "encoder_input_length", 
                      "decoder_inputs", "target"])

        self.set_gpu(self.gpu)

    @property
    def encoder_inputs(self):
        return self.encoder_inputs_

    @property
    def encoder_input_length(self):
        return self.encoder_input_length_ 

    @property
    def decoder_inputs(self):
        return self.decoder_inputs_

    @property
    def target(self):
        return self.target_

    @property
    def Batch(self):
        return self.Batch_

    @property
    def batch(self):
        return self.batch_

    @property
    def target_length(self):
        return self.target_length_

    def set_batch_buffers(self):
        if self.gpu < 0:
            name2buffer = self.name2cpu_buffer_
        else:
            name2buffer = self.name2gpu_buffer_

        enc_batch = [Variable(name2buffer[name])
                     for _, name in self.encoder_inputs]
        enc_len_batch = Variable(name2buffer["encoder_input_length"])
        dec_batch = [Variable(name2buffer[name])
                     for _, name in self.decoder_inputs]
        tgt_batch = Variable(name2buffer["target"])

        self.batch_ = self.Batch(
            enc_batch, enc_len_batch, dec_batch, tgt_batch)
            
    def get_batch(self):
        return self.batch



class DatasetOld(object):

    def __init__(self, input_data, target_data, sorting_length):

        self.batch_size_ = 1
        self.chunk_size_ = 500
        self.gpu_ = -1

        self.sorting_length_ = sorting_length
        
        self.data_ = []
        self.length_ = []
        self.cpu_buffers_ = []
        self.max_len_buf_ = torch.LongTensor()
        self.names_ = []

        for input, length, name in input_data:
            def getter(self):
                return input
            setattr(Dataset, name, property(getter))
            self.data_.append(input)
            self.length_.append(length)
            self.cpu_buffers_.append(input.new())
            self.names_.append(name)

        for target, length, name in target_data:
            def getter(self):
                return target
            setattr(Dataset, name, property(getter))
            self.data_.append(target)
            self.length_.append(length)
            self.cpu_buffers_.append(target.new())
            self.names_.append(name)
        
        self.Batch = namedtuple("Batch", self.names_)

    def set_gpu(self, gpu):
        self.gpu_ = gpu
        if self.gpu_ > -1:
            self.gpu_buffers_ = []
            for cpu_buffer in self.cpu_buffers_:
                self.gpu_buffers_.append(cpu_buffer.new().cuda(self.gpu_))

    def set_batch_size(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise Exception("Batch size must be a positive integer.")
        self.batch_size_ = batch_size
    
    @property
    def sorting_length(self):
        return self.sorting_length_

    @property
    def batch_size(self):
        return self.batch_size_

    @property
    def chunk_size(self):
        return self.chunk_size_

    @property
    def size(self):
        return self.sorting_length.size(0)

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
                
            buffers = self.cpu_buffers_ if self.gpu_ < 0 else self.gpu_buffers_
            variables = [Variable(buffer) for buffer in buffers]
            batch = self.Batch(*variables)

            yield batch





LMDatasetBatch = namedtuple("LMDatasetBatch", ["input", "output", "length"])

class LMDataset(object):

    def __init__(self, input, output, length):
        self.input_ = input
        self.output_ = output
        self.length_ = length

        self.batch_size_ = 1
        self.chunk_size_ = 500

        self.gpu_ = -1

        self.buf1 = torch.LongTensor()
        self.buf2 = torch.LongTensor()
        self.buf3 = torch.LongTensor()

    def set_gpu(self, gpu):
        self.gpu_ = gpu
        if self.gpu_ > -1:
            with torch.cuda.device(self.gpu_):
                self.cbuf1 = torch.cuda.LongTensor()
                self.cbuf2 = torch.cuda.LongTensor()
                self.cbuf3 = torch.cuda.LongTensor()

    def set_batch_size(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise Exception("Batch size must be a positive integer.")
        self.batch_size_ = batch_size
    
    @property
    def input(self):
        return self.input_

    @property
    def output(self):
        return self.output_

    @property
    def length(self):
        return self.length_

    @property
    def batch_size(self):
        return self.batch_size_

    @property
    def chunk_size(self):
        return self.chunk_size_

    @property
    def size(self):
        return self.input.size(0)

    def iter_batch(self):

        indices = [i for i in range(self.size)]
        random.shuffle(indices)
        indices = torch.LongTensor(indices)
        
        for i in range(0, self.size, self.batch_size * self.chunk_size):
            indices_chunk = indices[i:i + self.batch_size * self.chunk_size]
            lengths_batch = self.length.index_select(0, indices_chunk)
            sorted, sort_indices = torch.sort(
                lengths_batch, dim=0, descending=True)
            indices_sorted_ = indices_chunk.index_select(0, sort_indices)
            indices[i:i + self.batch_size * self.chunk_size] = indices_sorted_

        for i in range(0, self.size, self.batch_size):
            indices_batch = indices[i:i + self.batch_size]

            # THIS WONT WORK IF NOT SORTED
            max_len = self.length[indices_batch[0]]

            input_b = self.input.index_select(
                0, indices_batch, out=self.buf1)
            input_b = input_b[:,:max_len]

            output_b = self.output.index_select(
                0, indices_batch, out=self.buf2)
            output_b = output_b[:,:max_len]

            length_b = self.length.index_select(
                0, indices_batch, out=self.buf3)

            if self.gpu_ > -1:

                input_b = self.cbuf1.resize_(input_b.size()).copy_(input_b)
                output_b = self.cbuf2.resize_(output_b.size()).copy_(output_b)
                length_b = self.cbuf3.resize_(length_b.size()).copy_(length_b)

            yield LMDatasetBatch(
                Variable(input_b), Variable(output_b), Variable(length_b))
