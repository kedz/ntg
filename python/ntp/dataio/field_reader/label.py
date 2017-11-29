from .field_reader_base import FieldReaderBase
from ..vocabulary import Vocabulary

import torch


class Label(FieldReaderBase):

    def __init__(self, field, zero_indexing=True, vector_type=int, 
                 vocabulary=None):
        super(Label, self).__init__(field)
        
        self.vector_type = vector_type
        self.register_data("label_data")
        self.vocab_ = Vocabulary(
            zero_indexing=zero_indexing, special_tokens=vocabulary)
        self.labels_ = None

    def read_extract(self, data):
        if not isinstance(data, str):
            label_string = str(data)
        else:
            label_string = data

        label_index = self.vocab[label_string]
        if label_index is None:
            raise Exception(
                "Found unknown label string: {}".format(label_string))
        self.label_data.append(label_index)
        
    def fit_parameters(self):
        self.vocab.freeze()
        self.labels_ = tuple([token for token in self.vocab])
    
    def finalize_saved_data(self):
        if self.vector_type == int:
            data = (torch.LongTensor(self.label_data),)
        elif self.vector_type == float:
            data = (torch.FloatTensor(self.label_data),)
        else:
            data = (torch.ByteTensor(self.label_data),)
        return data

    @property
    def labels(self):
        if self.vocab.frozen:
            if self.labels_ is None:
                self.labels_ = tuple([token for token in self.vocab])
            return self.labels_
        else:
            return tuple([token for token in self.vocab])

    @property
    def vocab(self):
        return self.vocab_

    @property
    def zero_indexing(self):
        return self.vocab.zero_indexing

    @property
    def vector_type(self):
        return self.vector_type_

    @vector_type.setter
    def vector_type(self, vector_type):
        if vector_type not in (int, float, bytes):
            raise Exception("vector_type must be either int, float, or bytes.")
        self.vector_type_ = vector_type
