from abc import ABC, abstractmethod

class ReaderBase2(ABC):
    def __init__(self):
        super(ReaderBase2, self).__init__()
        self.data_attributes_ = []

    def register_data(self, name):
        self.data_attributes_.append(name)
        setattr(self, name, [])

    def reset_saved_data(self):
        for name in self.data_attributes_:
            setattr(self, name, [])
 
    @abstractmethod
    def read(self, datum):
        pass

    def fit_parameters(self):
        pass

    @abstractmethod
    def finalize_saved_data(self):
        pass

    def finish_read(self, reset=True):
        data = self.finalize_saved_data()
        if reset:
            self.reset_saved_data()
        return data


class DictFieldReaderWrapper(object):
    def __init__(self, reader, field):
        self.reader_ = reader
        self.field_ = field

    def read(self, datum):
        datum_field = datum[self.field_]
        self.reader_.read(datum_field)

    def __setitem__(self, key, value):
        self.reader_[key] = value

    def __getitem__(self, key):
        return self.reader_[key]

    def __getattr__(self, key):
        return getattr(self.reader_, key)


class IndexFieldReaderWrapper(object):
    def __init__(self, reader, field):
        if not isinstance(field, int) or field < 0:
            raise Exception("field must be non-negative integer.")
        self.reader_ = reader
        self.field_ = field

    def read(self, datum):
        datum_field = datum[self.field_]
        self.reader_.read(datum_field)

    def __setitem__(self, key, value):
        self.reader_[key] = value

    def __getitem__(self, key):
        return self.reader_[key]

    def __getattr__(self, key):
        return getattr(self.reader_, key)





class ReaderBase(ABC):
    def __init__(self, field, preprocessor, vocab):
        super(ReaderBase, self).__init__()

        if not isinstance(field, int) or field < 0:
            raise Exception("field argument must be a positive integer.")
         
        self.field_ = field
        self.preprocessor_ = preprocessor
        self.vocab_ = vocab
        self.data_attributes_ = []

    def register_data(self, name):
        self.data_attributes_.append(name)
        setattr(self, name, [])

    def reset(self):
        for name in self.data_attributes_:
            setattr(self, name, [])

    @property
    def preprocessor(self):
        return self.preprocessor_

    @property
    def vocab(self):
        return self.vocab_

    def freeze_vocab(self):
        self.vocab.freeze()

    def preprocess(self, data):
        return self.preprocessor.preprocess(data)

    @property
    def field(self):
        return self.field_

    def set_field(self, field):
        self.field_ = field
        return self
    
    def collect_stats(self, data):
        return self.process(data[self.field])
    
    @abstractmethod
    def save_data(self, data):
        pass

    def read(self, data):
        self.save_data(self.process(data[self.field]))

    @abstractmethod
    def process(self, string):
        pass

    @abstractmethod
    def info(self):
        pass
