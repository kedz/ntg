from .field_reader_base import FieldReaderBase
import torch

TENSOR_TYPES = (torch.LongTensor, torch.FloatTensor, torch.ByteTensor)

class Sequence(FieldReaderBase):

    def __init__(self, field_readers, field=None, sep=None, pad_value=-1):

        if field is None:
            super(Sequence, self).__init__(0)
            self.field_ = None
            self.field_type_ = type(None)
        else:
            super(Sequence, self).__init__(field)
        
        self.sep_ = sep
        self.pad_value_ = pad_value
        self.register_data("sequence_lengths")
        self.field_readers_ = tuple(field_readers)

    @property
    def sep(self):
        return self.sep_

    @property
    def pad_value(self):
        return self.pad_value_

    @property
    def field_readers(self):
        return self.field_readers_

    def read(self, raw_instance):
        if self.field is not None:
            if self.field_map is not None:
                sequence = raw_instance[self.field_map]
            else:
                sequence = raw_instance[self.field]

        else:
            sequence = raw_instance

        if self.sep is not None:
            sequence = sequence.split(self.sep)

        
        self.sequence_lengths.append(len(sequence))
        for item in sequence:
            self.read_extract(item)

    def read_extract(self, data):
        for field in self.field_readers:
            field.read(data)

    def fit_parameters(self):
        for field in self.field_readers:
            field.fit_parameters()


    def reset_saved_data(self):
        super(Sequence, self).reset_saved_data()
        for field in self.field_readers:
            field.reset_saved_data()

    def finalize_saved_data(self):
        all_results = []
        for field in self.field_readers:
            result_tuple = field.finalize_saved_data()
            seq_results = []
            for data in result_tuple:
                if isinstance(data, TENSOR_TYPES):
                    seq_results.append(self.pad_tensor_data(data))
                elif isinstance(data, (list, tuple)):
                    seq_results.append(self.pad_tuple_data(data))
                else:
                    raise Exception("Sequence does not know how to deal " \
                                    "with type {}".format(type(data)))
            all_results.append(tuple(seq_results))
        return tuple(all_results), torch.LongTensor(self.sequence_lengths)
            
    def pad_tensor_data(self, data):
        max_length = max(self.sequence_lengths)
        num_seqs = len(self.sequence_lengths)

        seq_data_dims = [num_seqs, max_length]
        seq_data_dims += [size for size in data.size()[1:]]
        seq_data = data.new(*seq_data_dims).fill_(self.pad_value)

        position = 0
        for i, size in enumerate(self.sequence_lengths):
            next_position = position + size
            seq_data[i][:size].copy_(data[position:next_position])
            position = next_position

        return seq_data

    def pad_tuple_data(self, data):
        max_length = max(self.sequence_lengths)
        num_seqs = len(self.sequence_lengths)
        seq_data = []
        position = 0
        for i, size in enumerate(self.sequence_lengths):
            next_position = position + size
            seq_data.append(tuple(data[position:next_position]))
            position = next_position
        return tuple(seq_data)
