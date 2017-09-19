import random
import torch
from torch.autograd import Variable
from text_preprocessor import VocabPreprocessor

DEFAULT_MAX_VOCAB_SIZE=80000

class Vocab(object):
    def __init__(self, preprocessor, at_least=1, top_k=DEFAULT_MAX_VOCAB_SIZE,
                 special_tokens=None, unknown_token=None):

        if not isinstance(preprocessor, VocabPreprocessor):
            raise Exception("First arg must be a VocabPreprocessor object.")

        self.pp_ = preprocessor.freeze()
        self.top_k_ = top_k
        self.at_least_ = at_least

        idx2token, token2idx = self.get_vocab_from_counts_(self.pp_.counts())
        
        if special_tokens is not None:
            for t in special_tokens:
                token2idx[t] = len(idx2token)
                idx2token.append(t)
        
        if unknown_token is not None:
            self.unk_idx_ = len(idx2token)
            token2idx[unknown_token] = self.unk_idx_
            idx2token.append(unknown_token)
        else:
            self.unk_idx_ = None

        self.idx2token_ = idx2token
        self.token2idx_ = token2idx

    @property
    def top_k(self):
        return self.top_k_

    @property
    def at_least(self):
        return self.at_least_

    def get_vocab_from_counts_(self, counts):
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        idx2tokens = [token for token, count in counts[:self.top_k] 
                      if count >= self.at_least]
        
        # 0 is the pad token and not used.
        idx2tokens = ["_PAD_"] + idx2tokens

        token2idx = {token: index for index, token in enumerate(idx2tokens)}

        return idx2tokens, token2idx

    def preprocess_lookup(self, items):
        tokens = self.pp_.preprocess(items)
        indices = self.lookup_tokens(tokens)
        return indices

    @property
    def unknown_index(self):
        return self.unk_idx_

    def index(self, token):
        return self.token2idx_.get(token, self.unknown_index)

    def lookup_tokens(self, tokens):
        return [self.index(token) for token in tokens]

    def token(self, index):
        return self.idx2token_[index]

    def inv_lookup_matrix(self, matrix):
        rows = matrix.size(0)
        cols = matrix.size(1)
        result = []
        for r in range(rows):
            row = [self.token(int(matrix[r,c])) for c in range(cols)
                   if matrix[r,c] > 0]
            result.append(row)
        return result

    @property
    def size(self):
        return len(self.idx2token_)

class LengthVocab(Vocab):
    def __init__(self, preprocessor, at_least=1, top_k=DEFAULT_MAX_VOCAB_SIZE,
                 special_tokens=None, unknown_token=None, sparse=True):

        if not isinstance(preprocessor, VocabPreprocessor):
            raise Exception("First arg must be a VocabPreprocessor object.")

        self.pp_ = preprocessor.freeze()
        self.top_k_ = top_k
        self.at_least_ = at_least
        self.sparse_ = sparse

        idx2token, token2idx = self.get_vocab_from_counts_(self.pp_.counts())
        
        if special_tokens is not None:
            for t in special_tokens:
                token2idx[t] = len(idx2token)
                idx2token.append(t)
        
        if unknown_token is not None:
            self.unk_idx_ = len(idx2token)
            token2idx[unknown_token] = self.unk_idx_
            idx2token.append(unknown_token)
        else:
            self.unk_idx_ = None

        self.idx2token_ = idx2token
        self.token2idx_ = token2idx

    def get_vocab_from_counts_(self, counts):
        counts = sorted(counts.items(), key=lambda x: int(x[0]))
        idx2tokens = [token for token, count in counts[:self.top_k] 
                      if count >= self.at_least]

        # 0 is the pad token and not used.
        idx2tokens = ["_PAD_"] + idx2tokens

        token2idx = {token: index for index, token in enumerate(idx2tokens)}

        return idx2tokens, token2idx




class Seq2SeqDataset(object):
    def __init__(self, source, source_length, target_in, target_out, 
                 target_length, target_length_toks):
        self.source_ = source
        self.source_length_ = source_length
        
        self.target_in_ = target_in
        self.target_out_ = target_out
        self.target_length_ = target_length
        self.target_length_toks_ = target_length_toks

        self.batch_size_ = 2
        self.chunk_size_ = 10

        self.gpu_ = 0

        self.buf1 = torch.LongTensor()
        self.buf2 = torch.LongTensor()
        self.buf3 = torch.LongTensor()
        self.buf4 = torch.LongTensor()
        self.buf5 = torch.LongTensor()
        self.buf6 = torch.LongTensor()

    def set_gpu(self, gpu):
        self.gpu_ = gpu
        if self.gpu_ > -1:
            with torch.cuda.device(self.gpu_):
                self.cbuf1 = torch.cuda.LongTensor()
                self.cbuf2 = torch.cuda.LongTensor()
                self.cbuf3 = torch.cuda.LongTensor()
                self.cbuf4 = torch.cuda.LongTensor()
                self.cbuf5 = torch.cuda.LongTensor()
                self.cbuf6 = torch.cuda.LongTensor()

    def set_batch_size(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise Exception("Batch size must be a positive integer.")
        self.batch_size_ = batch_size
    
    @property
    def source(self):
        return self.source_

    @property
    def source_length(self):
        return self.source_length_

    @property
    def target_in(self):
        return self.target_in_

    @property
    def target_out(self):
        return self.target_out_

    @property
    def target_length(self):
        return self.target_length_

    @property
    def batch_size(self):
        return self.batch_size_

    def set_batch_size(self, batch_size):
        self.batch_size_ = batch_size
    
    @property
    def chunk_size(self):
        return self.chunk_size_

    @property
    def size(self):
        return self.source_.size(0)

    def iter_batch(self):

        indices = [i for i in range(self.size)]
        random.shuffle(indices)
        indices = torch.LongTensor(indices)
        
        for i in range(0, self.size, self.batch_size * self.chunk_size):
            indices_chunk = indices[i:i + self.batch_size * self.chunk_size]
            lengths_batch = self.source_length.index_select(0, indices_chunk)
            sorted, sort_indices = torch.sort(
                lengths_batch, dim=0, descending=True)
            indices_sorted_ = indices_chunk.index_select(0, sort_indices)
            indices[i:i + self.batch_size * self.chunk_size] = indices_sorted_

        for i in range(0, self.size, self.batch_size):
            indices_batch = indices[i:i + self.batch_size]

            # THIS WONT WORK IF NOT SORTED
            max_src_len = self.source_length[indices_batch[0]]

            src_b = self.source.index_select(0, indices_batch, out=self.buf1)
            src_b = src_b[:,:max_src_len]
            src_len_b = self.source_length.index_select(
                0, indices_batch, out=self.buf2)
            tgt_len_b = self.target_length.index_select(
                0, indices_batch, out=self.buf5)
            max_tgt_len = torch.max(tgt_len_b)

            tgt_in_b = self.target_in[:,:max_tgt_len].index_select(
                0, indices_batch, out=self.buf3)
            tgt_out_b = self.target_out[:,:max_tgt_len].t().index_select(
                1, indices_batch, out=self.buf4)

            if isinstance(self.target_length_toks_, torch.LongTensor):
                tgt_lt_b = self.target_length_toks_.index_select(
                0, indices_batch, out=self.buf6)
            else:
                tgt_lt_b = None

            if self.gpu_ > -1:

                src_b = self.cbuf1.resize_(src_b.size()).copy_(src_b)
                src_len_b = self.cbuf2.resize_(src_len_b.size()).copy_(
                    src_len_b)
                tgt_in_b = self.cbuf3.resize_(tgt_in_b.size()).copy_(tgt_in_b)
                tgt_out_b = self.cbuf4.resize_(tgt_out_b.size()).copy_(
                    tgt_out_b)
                tgt_len_b = self.cbuf5.resize_(tgt_len_b.size()).copy_(
                    tgt_len_b)
                if isinstance(self.target_length_toks_, torch.LongTensor):
                    tgt_lt_b = self.cbuf6.resize_(tgt_lt_b.size()).copy_(
                        tgt_lt_b)


            batch = Seq2SeqBatch(
                src_b, src_len_b, tgt_in_b, tgt_out_b.t(), tgt_len_b, tgt_lt_b)
            yield batch


class Seq2SeqBatch(object):
    def __init__(self, source, source_length, target_in, target_out, 
                 target_length, tgt_lt_b=None):
        self.source_ = Variable(source)
        self.source_length_ = source_length
        self.target_in_ = Variable(target_in)
        self.target_out_ = Variable(target_out)
        self.target_length_ = target_length
        if isinstance(tgt_lt_b, torch.LongTensor):
            self.target_len_tok_ = Variable(tgt_lt_b)
        elif isinstance(tgt_lt_b, torch.cuda.LongTensor):
            self.target_len_tok_ = Variable(tgt_lt_b)
        else:
            self.target_len_tok_ = None

    @property
    def source(self):
        return self.source_

    @property
    def source_length(self):
        return self.source_length_

    @property
    def target_in(self):
        return self.target_in_

    @property
    def target_out(self):
        return self.target_out_

    @property
    def target_length(self):
        return self.target_length_
