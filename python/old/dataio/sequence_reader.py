import torch

class DiscreteSequenceReader(object):

    def __init__(self, field, vocab, 
                 left_pad=None, right_pad=None, 
                 offset_io_pair=True):

        if left_pad is not None:
            if isinstance(left_pad, (tuple, list)):
                left_pad = tuple([vocab.index(p) for p in left_pad])
            else:
                left_pad = tuple([vocab.index(left_pad)])
                
        if right_pad is not None:
            if isinstance(right_pad, (tuple, list)):
                right_pad = tuple([vocab.index(p) for p in right_pad])
            else:
                right_pad = tuple([vocab.index(right_pad)])
            
        self.offset_io_pair_ = offset_io_pair

        self.left_pad_ = left_pad
        self.right_pad_ = right_pad
        self.field_ = field
        self.vocab_ = vocab

        if self.offset_io_pair:
            self.data_in_ = []
            self.data_out_ = []
        else:
            self.data_ = []
        self.data_length_ = []


    @property
    def vocab(self):
        return self.vocab_

    @property
    def field(self):
        return self.field_

    @property
    def left_pad(self):
        return self.left_pad_

    @property
    def right_pad(self):
        return self.right_pad_

    @property
    def offset_io_pair(self):
        return self.offset_io_pair_

    def reset(self):
        if self.offset_io_pair:
            self.data_in_ = []
            self.data_out_ = []
        else:
            self.data_ = []
        self.data_length_ = []

    def read(self, items):
        tokens = tuple(self.vocab.preprocess_lookup(items))
        if self.left_pad is not None:
            tokens = self.left_pad + tokens
        if self.right_pad is not None:
            tokens = tokens + self.right_pad
        if self.offset_io_pair:
            self.data_in_.append(tokens[:-1])
            self.data_out_.append(tokens[1:])
            self.data_length_.append(len(tokens) - 1)
        else:
            self.data_.append(tokens)
            self.data_length_.append(len(tokens))

    def finish(self, reset=True):
        
        data_size = len(self.data_length_)
        max_len = max(self.data_length_)
        zed = tuple([0])

        for i in range(data_size):
            if self.data_length_[i] < max_len:
                if self.offset_io_pair:
                    self.data_in_[i] += zed * (max_len - self.data_length_[i])
                    self.data_out_[i] += zed * (max_len - self.data_length_[i])

                else:
                    self.data_[i] += zed * (max_len - self.data_length_[i])

        if self.offset_io_pair:        
            input = torch.LongTensor(self.data_in_)
            output = torch.LongTensor(self.data_out_)
            length = torch.LongTensor(self.data_length_)
            finished_data = (input, output, length)
       
        else:
            data = torch.LongTensor(self.data_)
            length = torch.LongTensor(self.data_length_)
            finished_data = (data, length)

        if reset:
            self.reset()

        return finished_data
