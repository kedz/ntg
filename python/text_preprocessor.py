from collections import defaultdict
import pandas as pd
import re

class TextPreprocessor(object):

    def __init__(self, tokenizer=None, lower=True, replace_digits=True):
        
        if tokenizer is None:
            tokenizer = lambda x: x.split()
        
        self.tokenizer_ = tokenizer
        self.lower_ = lower
        self.replace_digits_ = replace_digits
        
    def preprocess(self, line):
        if self.lower_:
            line = line.lower()
        if self.replace_digits_:
            line = re.sub(r"\d", "D", line)
        return self.tokenizer_(line)


class VocabPreprocessor(TextPreprocessor):
    def __init__(self, field):
        super(VocabPreprocessor, self).__init__()
        self.field_ = field
        self.counts_ = defaultdict(int)
        self.total_lines_ = 0
        self.is_frozen_ = False

    def preprocess(self, items):
        tokens = super(VocabPreprocessor, self).preprocess(items[self.field_])
        if not self.frozen():
            for token in tokens:
                self.counts_[token] += 1
            self.total_lines_ += 1
        return tokens

    def counts(self):
        return self.counts_

    def __str__(self):
       data = sorted(self.counts_.items(), key=lambda x: x[1], reverse=True)
       df = pd.DataFrame(data, columns=["token", "frequency"])
       return str(df)

    def freeze(self, is_frozen=True):
        self.is_frozen_ = is_frozen
        return self

    def frozen(self):
        return self.is_frozen_
