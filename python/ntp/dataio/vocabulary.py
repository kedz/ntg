import sys
   

DEFAULT_TOPK = sys.maxsize

class Vocabulary(object):
    def __init__(self, top_k=DEFAULT_TOPK, at_least=1, unknown_token=None,
                 special_tokens=None, zero_indexing=True):
        self.top_k = top_k
        self.at_least = at_least

        self.count_ = {}
        self.token2index_ = {}
        self.index2token_ = []
        self.frozen_ = False
        self.zero_indexing_ = zero_indexing
        if not self.zero_indexing:
            self.index2token_.append(None)

        self.unknown_token_ = unknown_token
        if self.unknown_token is not None:
            self.unknown_index_ = len(self.index2token_)
            self.token2index_[self.unknown_token] = len(self.index2token_)
            self.index2token_.append(self.unknown_token)
        else:
            self.unknown_index_ = None

        if special_tokens is None:
            self.special_tokens_ = ()
        elif isinstance(special_tokens, (str, list, tuple)):
            if isinstance(special_tokens, str):
                special_tokens = [special_tokens]
            self.special_tokens_ = tuple(special_tokens)
            
            for token in self.special_tokens:
                self.token2index_[token] = len(self.index2token_)
                self.index2token_.append(token)
            
    def __getitem__(self, item):
        
        if isinstance(item, str):
            if self.frozen:
                return self.token2index_.get(item, self.unknown_index)
            else:
                self.count_[item] = self.count_.get(item, 0) + 1
                if item not in self.token2index_:
                    index = len(self.index2token_)
                    self.token2index_[item] = index
                    self.index2token_.append(item)
                    return index

                return self.token2index_[item]
        else:
            return self.index2token_[item]

    def freeze(self):
        
        if self.frozen:
            return 

        index2token = [(k, v) for k, v in self.count_.items() 
                       if v >= self.at_least 
                       and k not in self.special_tokens
                       and k != self.unknown_token]

        index2token.sort(key=lambda x: x[1], reverse=True)
        index2token = [k for k, v in index2token[:self.top_k]]
        
        index2token = list(self.special_tokens) + index2token
        
        if self.unknown_token is not None:
            index2token = [self.unknown_token] + index2token

        if not self.zero_indexing:
            index2token = [None] + index2token

        token2index = {t: i for i, t in enumerate(index2token)}

        if self.unknown_token is not None:
            self.unknown_index_ = token2index[self.unknown_token]
        else:
            self.unknown_index_ = None
        
        self.index2token_ = index2token
        self.token2index_ = token2index        
        self.frozen_ = True

    @property
    def frozen(self):
        return self.frozen_

    @property
    def zero_indexing(self):
        return self.zero_indexing_

    @property
    def unknown_token(self):
        return self.unknown_token_

    @property
    def unknown_index(self):
        return self.unknown_index_

    @property
    def special_tokens(self):
        return self.special_tokens_

    @property
    def top_k(self):
        return self.top_k_

    @top_k.setter
    def top_k(self, top_k):
        if not isinstance(top_k, int) or top_k < 1:
            raise Exception("top_k must be a positive int.")
        self.top_k_ = top_k

    @property
    def at_least(self):
        return self.at_least_

    @at_least.setter
    def at_least(self, at_least):
        if not isinstance(at_least, int) or at_least < 1:
            raise Exception("at_least must be a positive int.")
        self.at_least_ = at_least

    @property
    def size(self):
        return len(self.index2token_) - (1 - int(self.zero_indexing))

    def __len__(self):
        return self.size

    def __str__(self):
        return "Vocabulary(size={})".format(self.size)

    def __iter__(self):
        if self.zero_indexing:
            for token in self.index2token_:
                yield token
        else:
            for token in self.index2token_[1:]:
                yield token

    def contains(self, token):
        return self.token2index_.get(token, None) is not None
