import torch
from dataio.reader.reader_base import ReaderBase
from preprocessor import SimplePreprocessor
from vocab import Vocab


class DiscreteFeatureReader(ReaderBase):
    def __init__(self, field=0, missing_token="MISSING"):

        v = Vocab(zero_indexing=False, special_tokens=[missing_token])
        pp = SimplePreprocessor(strip=True, lowercase=False)
        
        super(DiscreteFeatureReader, self).__init__(field, pp, v)

        self.register_data("data_")
 
    def process(self, string):
        string = self.preprocess(string)
        return self.vocab.index(string)
    
    def save_data(self, datum):
        self.data_.append(datum)

    def info(self):
        total = sum(v for k, v in self.vocab.count.items())
        unique = len(self.vocab.count)
        msg = "DiscreteFeatureReader found {} instances and {} " \
            "unique feature values.\n".format(
            total, unique)
        msg += "Feature (Counts)\n"
        for label, count in self.vocab.count.items():
            msg += "{} {} ({})\n".format(self.vocab.index(label), label, count)
        return msg
                
    def finish(self, reset=True):
        finished_data = (torch.LongTensor(self.data_),)
        if reset:
            self.reset()
        return finished_data
