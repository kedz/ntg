import re
from preprocessor.simple_preprocessor import SimplePreprocessor

def _tokenize(tokens):
    return tokens.split()

class TextPreprocessor(SimplePreprocessor):
    def __init__(self, strip=True, lowercase=False, replace_digits=True,
                 tokenizer=None):

        super(TextPreprocessor, self).__init__(
            strip=strip, lowercase=lowercase)
        self.replace_digits_ = replace_digits 
        self.tokenizer_ = _tokenize if tokenizer is None else tokenizer

    @property
    def replace_digits(self):
        return self.replace_digits_

    @property
    def tokenize(self):
        return self.tokenizer_

    def set_tokenizer(self, tokenizer):
        self.tokenizer_ = tokenizer

    def set_replace_digits(self, replace_digits):
        self.replace_digits_ = replace_digits

    def preprocess(self, string):
        string = super(TextPreprocessor, self).preprocess(string)
        if self.replace_digits: 
            string = re.sub(r"\d", "D", string)
        return self.tokenize(string)
