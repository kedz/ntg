import torch
from dataio.reader.reader_base import ReaderBase2, DictFieldReaderWrapper

class DenseRealVectorReader(ReaderBase2):

    @staticmethod
    def field_reader(field_type, field_name, vector_size, sep=None):
        if field_type == "dict":
            drvr = DenseRealVectorReader(vector_size, sep=sep)
            return DictFieldReaderWrapper(drvr, field_name)
        else:
            raise Exception("field_type invalid.")

    def __init__(self, vector_size, sep=None):
        if not isinstance(vector_size, int) or vector_size < 1:
            raise Exception("vector_size must be a positive integer.")
        super(DenseRealVectorReader, self).__init__()
        self.vector_size_ = vector_size
        self.sep_ = sep
        self.register_data("vectors")

    def read(self, datum):
        if self.sep_ is None:
            vec = datum
        else:
            vec = [float(x) for x in datum.split(self.sep_)]
        if len(vec) != self.vector_size_:
            raise Exception(
                "Incompatible vector size: found {}, expected {}".format(
                    len(vec), self.vector_size_))
        self.vectors.append(vec)
    
    def finalize_saved_data(self):
        data = (torch.FloatTensor(self.vectors),)
        return data


