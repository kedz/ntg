from abc import ABC, abstractmethod 


class FileReaderBase(ABC):
    def __init__(self, readers, verbose=False):
        super(ABC, self).__init__() 
        
        self.readers_ = tuple(readers)
        self.verbose = verbose

    @property
    def readers(self):
        return self.readers_

    def fit_parameters(self, paths=None):
        if isinstance(paths, (list, tuple)):
            for path in paths:
                self.apply_readers(path) 
        elif paths is not None:
            self.apply_readers(paths) 

        for reader in self.readers_:
            reader.fit_parameters()
            reader.reset_saved_data()
    
    @abstractmethod
    def apply_readers(self, path):
        pass

    def read(self, path):
        self.apply_readers(path)
        all_reader_data = []
        for reader in self.readers_:
            all_reader_data.append(reader.finish_read())
        return tuple(all_reader_data)
