from abc import ABC, abstractmethod

class PreprocessorBase(ABC):
    def __init__(self):
        super(PreprocessorBase, self).__init__()

    @abstractmethod
    def preprocess(self, data):
        pass
