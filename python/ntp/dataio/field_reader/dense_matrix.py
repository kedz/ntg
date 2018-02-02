from .field_reader_base import FieldReaderBase
import torch

class DenseMatrix(FieldReaderBase):
    def __init__(self, field, expected_size_dim1="any", 
                 expected_size_dim2="any", matrix_type=float,
                 pad_value=-1):
        """
            expected sizes:
                any: use pad value to make a padded 3d tensor
                first: use first read value and make sure it is consistent
                a pos int: make sure this dimension is always the value
                of the integer.
        """


        super(DenseMatrix, self).__init__(field)

        # TODO add row and column sep
        #self.sep_ = sep
        self.expected_size_dim1 = expected_size_dim1
        self.expected_size_dim2 = expected_size_dim2
        self.matrix_type_ = matrix_type
        self.register_data("matrices")

        if self.expected_size_dim1 == "any":
            self.register_data("dim1_sizes")
        if self.expected_size_dim2 == "any":
            self.register_data("dim2_sizes")

        self.pad_value = pad_value
        self.sep = None

    @property
    def expected_size_dim1(self):
        return self.expected_size_dim1_

    @expected_size_dim1.setter
    def expected_size_dim1(self, new_size):
        if new_size in ["any", "first"]:
            self.expected_size_dim1_ = new_size
        elif isinstance(new_size, int) and new_size > 0:
            self.expected_size_dim1_ = new_size
        else:
            raise Exception(
                "Expected size must be 'any', 'first', or a positive int.")

    @property
    def expected_size_dim2(self):
        return self.expected_size_dim2_

    @expected_size_dim2.setter
    def expected_size_dim2(self, new_size):
        if new_size in ["any", "first"]:
            self.expected_size_dim2_ = new_size
        elif isinstance(new_size, int) and new_size > 0:
            self.expected_size_dim2_ = new_size
        else:
            raise Exception(
                "Expected size must be 'any', 'first', or a positive int.")

    @property
    def matrix_type(self):
        return self.matrix_type_

    @matrix_type.setter
    def matrix_type(self, matrix_type):
        if matrix_type not in (bytes, int, float):
            raise Exception("matrix_type must be bytes, int, long, or float.")
        self.matrix_type_ = matrix_type
         
    def read_extract(self, vector_or_string):
        # TODO make work with strings

        if self.sep is None:
            if isinstance(vector_or_string, str):
                raise Exception("DenseMatrix can't read strings yet.")
                #vector = [float(vector_or_string)]
            elif isinstance(vector_or_string, (int, float)):
                matrix = [[float(vector_or_string)]]
            else:
                matrix = vector_or_string
        else:
            raise Exception("DenseMatrix can't read strings yet.")
            #vector = [float(x) 
            #          for x in vector_or_string.split(self.sep)]

        matrix = self.make_tensor(matrix)
        
        if self.expected_size_dim1 == "first":
            self.expected_size_dim1 = matrix.size(0)
        elif self.expected_size_dim1 != "any":
            if self.expected_size_dim1 != matrix.size(0):
                raise Exception(
                    ("Found matrix of size ({}, {}) " \
                     "but expecting ({}, {}).").format(
                         matrix.size(0),
                         matrix.size(1),
                         self.expected_size_dim1,
                         self.expected_size_dim2))
        else:
            self.dim1_sizes.append(matrix.size(0))

        if self.expected_size_dim2 == "first":
            self.expected_size_dim2 = matrix.size(1)
        elif self.expected_size_dim2 != "any":
            if self.expected_size_dim2 != matrix.size(1):
                raise Exception(
                    ("Found matrix of size ({}, {}) " \
                     "but expecting ({}, {}).").format(
                         matrix.size(0),
                         matrix.size(1),
                         self.expected_size_dim1,
                         self.expected_size_dim2))
        else:
            self.dim2_sizes.append(matrix.size(1))

        self.matrices.append(matrix)

    def make_tensor(self, obj):
        if self.matrix_type == int:
            return torch.FloatTensor(obj).long()
        elif self.matrix_type == float:
            return torch.FloatTensor(obj)
        else:
            return torch.FloatTensor(obj).byte()

    def new_tensor(self, dims):
        if self.matrix_type == int:
            return torch.LongTensor(*dims)
        elif self.matrix_type == float:
            return torch.FloatTensor(*dims)
        else:
            return torch.ByteTensor(*dims)


    def finalize_saved_data(self):

        dims = [len(self.matrices)]
        if self.expected_size_dim1 == "any":
            dims.append(max(self.dim1_sizes))
        else:
            dims.append(self.matrices[0].size(0))

        if self.expected_size_dim2 == "any":
            dims.append(max(self.dim2_sizes))
        else:
            dims.append(self.matrices[0].size(1))

        result = self.new_tensor(dims).fill_(self.pad_value)

        for i, matrix in enumerate(self.matrices):
            size1 = matrix.size(0)
            size2 = matrix.size(1)
            result[i][:size1,:size2].copy_(matrix)

        if "any" in [self.expected_size_dim1, self.expected_size_dim2]:
            if self.expected_size_dim1 == "any":
                length1 = torch.LongTensor(self.dim1_sizes)
            else:
                length1 = torch.LongTensor([dims[1]] * dims[0])

            if self.expected_size_dim2 == "any":
                length2 = torch.LongTensor(self.dim2_sizes)
            else:
                length2 = torch.LongTensor([dims[2]] * dims[0])
            return (result, length1, length2)

        else:
            return (result,)
