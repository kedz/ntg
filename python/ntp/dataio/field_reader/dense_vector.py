from .field_reader_base import FieldReaderBase
import torch

class DenseVector(FieldReaderBase):
    def __init__(self, field, sep=None, expected_size="first", 
                 vector_type=float, pad_value=-1):
        super(DenseVector, self).__init__(field)
        self.expected_size = expected_size
        self.sep_ = sep
        self.vector_type_ = vector_type
        self.register_data("vectors")
        if self.expected_size == "any":
            self.register_data("lengths")
        self.pad_value = pad_value

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

    @expected_size.setter
    def expected_size(self, new_size):
        if new_size in ["any", "first"]:
            self.expected_size_ = new_size
        elif isinstance(new_size, int) and new_size > 0:
            self.expected_size_ = new_size
        else:
            raise Exception(
                "Expected size must be 'any', 'first', or a positive int.")

    def read_extract(self, vector_or_string):
        if self.sep is None:
            if isinstance(vector_or_string, str):
                vector = [float(vector_or_string)]
            elif isinstance(vector_or_string, (int, float)):
                vector = [float(vector_or_string)]
            else:
                vector = vector_or_string
        else:
            vector = [float(x) 
                      for x in vector_or_string.split(self.sep)]

        if self.expected_size == "first":
            self.expected_size = len(vector)
        elif self.expected_size != "any":
            if self.expected_size != len(vector):
                raise Exception(
                    ("Found vector of size {} " \
                     "but expecting {}.").format(
                         len(vector),
                         self.expected_size))
        else:
            self.lengths.append(len(vector))
        self.vectors.append(vector)

    def make_tensor(self, obj):
        if self.vector_type == int:
            return torch.FloatTensor(obj).long()
        elif self.vector_type == float:
            return torch.FloatTensor(obj)
        else:
            return torch.FloatTensor(obj).byte()

    def new_tensor(self, dims):
        if self.vector_type == int:
            return torch.LongTensor(*dims)
        elif self.vector_type == float:
            return torch.FloatTensor(*dims)
        else:
            return torch.ByteTensor(*dims)

    def finalize_saved_data(self):

        if self.expected_size == "any":
            dims = [len(self.vectors), max(self.lengths)]
            lengths = torch.LongTensor(self.lengths)
            result = self.new_tensor(dims).fill_(self.pad_value)

            for i, vec in enumerate(self.vectors):
                size = lengths[i]
                result[i,:size].copy_(self.make_tensor(vec))
            return (result, lengths)

        else:
            return (self.make_tensor(self.vectors),)
