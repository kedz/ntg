import torch
from dataio.reader.reader_base import ReaderBase, ReaderBase2, DictFieldReaderWrapper
from preprocessor import SimplePreprocessor
from vocab import Vocab


class LabelReader2(ReaderBase2):

    @staticmethod
    def field_reader(field_type, field_name, strip=True, lowercase=False):
        if field_type == "dict":
            lr = LabelReader2(strip=strip, lowercase=lowercase)
            return DictFieldReaderWrapper(lr, field_name)
        else:
            raise Exception("field_type invalid.")

    def __init__(self, strip=True, lowercase=False):
        super(LabelReader2, self).__init__()
        self.vocab_ = Vocab(zero_indexing=True)
        self.preprocessor_ = SimplePreprocessor(
            strip=strip, lowercase=lowercase)
        self.register_data("labels")

    def read(self, datum):
        label_string = self.preprocessor_.preprocess(datum)
        label_index = self.vocab_.index(label_string)
        self.labels.append(label_index)
        
    def fit_parameters(self):
        self.vocab_.freeze()
    
    def finalize_saved_data(self):
        data = (torch.LongTensor(self.labels),)
        return data

    @property
    def vocab(self):
        return self.vocab_


class LabelReader(ReaderBase):
    def __init__(self, field=0, strip=True, lowercase=True):

        v = Vocab(zero_indexing=True)
        pp = SimplePreprocessor(strip=strip, lowercase=lowercase)
        
        super(LabelReader, self).__init__(field, pp, v)

        self.register_data("data_")
        
    @property
    def labels(self):
        return [self.vocab.token(idx) for idx in range(self.vocab.size)]

    def process(self, string):
        string = self.preprocess(string)
        return self.vocab.index(string)
    
    def save_data(self, datum):
        self.data_.append(datum)

    def info(self):
        total = sum(v for k, v in self.vocab.count.items())
        unique = len(self.vocab.count)
        msg = "LabelReader found {} instances and {} unique labels.\n".format(
            total, unique)
        msg += "Labels (Counts)\n"
        for label, count in self.vocab.count.items():
            msg += "{} {} ({})\n".format(self.vocab.index(label), label, count)
        return msg
                
    def finish(self, reset=True):
        finished_data = (torch.LongTensor(self.data_),)
        if reset:
            self.reset()
        return finished_data
