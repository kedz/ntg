from .file_reader_base import FileReaderBase
import re
import torch


class CONLLUReader(FileReaderBase):
    def __init__(self, readers, pad_value=0, verbose=False):
        super(CONLLUReader, self).__init__(readers, verbose=verbose)
        self.pad_value_ = 0

    @property
    def pad_value(self):
        return self.pad_value_

    def apply_readers(self, path):
        sizes = []
        with open(path, "r") as fp:
            in_ex = False
            example_size = 0

            for line in fp:
                if re.match(r"^\d+(\.\d+)?\t", line):
                    items = line.split("\t")[1:]
                    if in_ex == False:
                        example_size = 0
                        in_ex = True
                    example_size += 1
                    for reader in self.readers:
                        reader.read(items)

                elif in_ex == True:
                    in_ex = False
                    sizes.append(example_size)
            if in_ex:
                in_ex = False
                sizes.append(example_size)
        return sizes
    
    def read(self, path):
        lengths = self.apply_readers(path)
        lengths = torch.LongTensor(lengths)
        max_length = lengths.max()
        all_reader_data = []
        for reader in self.readers_:
            tensors = reader.finish_read()
            tensors = (self.pad_data(tensor, lengths, max_length)
                       for tensor in tensors)
            all_reader_data.append(tensors)
        all_reader_data.append(lengths)        

        return tuple(all_reader_data)

    def pad_data(self, tensor, lengths, max_length):
        dims = [lengths.size(0), max_length] + [x for x in tensor.size()][1:]
        pad_tensor = tensor.new(*dims).fill_(self.pad_value)
        pos = 0
        for i in range(lengths.size(0)):
            pad_tensor[i,:lengths[i]].copy_(tensor[pos:pos+lengths[i]])
            pos += lengths[i]
        return pad_tensor
