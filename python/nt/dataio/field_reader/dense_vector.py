from .field_reader_base import FieldReaderBase
import torch

class DenseVector(FieldReaderBase):
    def __init__(self, field, sep=None, expected_size=None, vector_type=float):
        super(DenseVector, self).__init__(field)
        self.expected_size_ = expected_size
        self.sep_ = sep
        self.vector_type_ = vector_type
        self.register_data("vectors")

    @property
    def vector_type(self):
        return self.vector_type_

    @vector_type.setter
    def vector_type(self, vector_type):
        if vector_type not in (bytes, int, float):
            raise Exception("vector_type must be bytes, int, long, or float.")
        self.vector_type_ = vector_type
         
    @property
    def sep(self):
        return self.sep_

    @property
    def expected_size(self):
        return self.expected_size_

    def read_extract(self, vector_or_string):
        if self.sep is None:
            vector = vector_or_string
        else:
            vector = [self.vector_type(x) 
                      for x in vector_or_string.split(self.sep)]
        if self.expected_size is None:
            self.expected_size_ = len(vector)
        
        if len(vector) != self.expected_size:
            raise Exception("Found vector of size {} but expecting {}".format(
                len(vector), self.expected_size))

        self.vectors.append(vector)

    def finalize_saved_data(self):
        if self.vector_type == int:
            data = (torch.FloatTensor(self.vectors).long(),)
        elif self.vector_type == float:
            data = (torch.FloatTensor(self.vectors),)
        else:
            data = (torch.FloatTensor(self.vectors).byte(),)
        return data
