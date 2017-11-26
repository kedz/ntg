from abc import ABC, abstractmethod

class OptimizerBase(ABC):
    def __init__(self):
        super(OptimizerBase, self).__init__()

    @abstractmethod
    def reset(self):
        pass
