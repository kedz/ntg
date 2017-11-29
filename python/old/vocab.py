
DEFAULT_TOPK=10000000

class Vocab(object):

    def __init__(self, top_k=DEFAULT_TOPK, at_least=1,
                 special_tokens=None, unknown_token=None,
                 zero_indexing=False):
        self.top_k_ = top_k
        self.at_least_ = at_least
        self.zero_indexing_ = zero_indexing
        self.is_frozen_ = False
        self.count_ = {}
        self.token2index_ = {}
        self.index2token_ = []

        if isinstance(special_tokens, list):
            special_tokens = tuple(special_tokens)
        elif isinstance(special_tokens, str):
            special_tokens = tuple([special_tokens])
        elif isinstance(special_tokens, type(None)):
            special_tokens = tuple([])
        elif not isinstance(special_tokens, tuple):
            raise Exception(
                "special_tokens must be str, list/tuple of strs, or None.")

        self.special_tokens_ = special_tokens
        self.unknown_token_ = unknown_token
        self.init_vocab()

    def init_vocab(self):

        index2token = [(k, v) for k, v in self.count.items() 
                       if v >= self.at_least 
                       and k not in self.special_tokens
                       and k != self.unknown_token]

        index2token.sort(key=lambda x: x[1], reverse=True)
        index2token = [k for k, v in index2token[:self.top_k]]
        
        index2token.extend(self.special_tokens)

        if self.has_unknown_token:
            index2token.append(self.unknown_token)

        if not self.zero_indexing:
            index2token = [None] + index2token

        token2index = {t: i for i, t in enumerate(index2token)}

        if self.has_unknown_token:
            self.unknown_index_ = token2index[self.unknown_token]
        else:
            self.unknown_index_ = None
        
        self.index2token_ = index2token
        self.token2index_ = token2index        

    @property
    def special_tokens(self):
        return self.special_tokens_

    @property
    def unknown_index(self):
        return self.unknown_index_

    @property
    def unknown_token(self):
        return self.unknown_token_

    @property
    def has_unknown_token(self):
        return self.unknown_token_ is not None

    @property
    def is_frozen(self):
        return self.is_frozen_
    
    @property
    def count(self):
        return self.count_

    @property
    def top_k(self):
        return self.top_k_

    @property
    def at_least(self):
        return self.at_least_

    @property
    def zero_indexing(self):
        return self.zero_indexing_

    def freeze(self):
        self.init_vocab()
        self.is_frozen_ = True

    def thawed_index_(self, token):
        self.count_[token] = self.count_.get(token, 0) + 1
        if token not in self.token2index_:
            self.token2index_[token] = len(self.index2token_)
            self.index2token_.append(token)
        return self.token2index_[token]

    def frozen_index_(self, token):
        if self.has_unknown_token:
            return self.token2index_.get(token, self.unknown_index)
        else:
            return self.token2index_[token]

    @property
    def index(self):
        if self.is_frozen:
            return self.frozen_index_
        else:
            return self.thawed_index_

    def token(self, index):
        return self.index2token_[index]

    @property
    def size(self):
        return len(self.index2token_)
