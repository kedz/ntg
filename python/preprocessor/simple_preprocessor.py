
from preprocessor.preprocessor_base import PreprocessorBase

class SimplePreprocessor(PreprocessorBase):
    def __init__(self, strip=True, lowercase=False):
        self.strip_ = strip
        self.lowercase_ = lowercase

    @property
    def strip(self):
        return self.strip_

    @property
    def lowercase(self):
        return self.lowercase_

    def set_strip(self, strip):
        self.strip_ = strip

    def set_lowercase(self, lowercase):
        self.lowercase_ = lowercase

    def preprocess(self, data):
        if self.strip:
            data = data.strip()
        if self.lowercase:
            data = data.lower()
        return data
