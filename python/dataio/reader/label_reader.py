import torch
from dataio.reader.reader_base import ReaderBase
from preprocessor import SimplePreprocessor
from vocab import Vocab

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
