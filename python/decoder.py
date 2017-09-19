import torch
import torch.nn as nn
import torch.nn.functional as F
import attention as attn
from torch.autograd import Variable

class RNNDecoder(nn.Module):

    def __init__(self, vocab, embedding_dim, rnn_type="gru", num_layers=3, 
                 hidden_dim=None, dropout=0.0,
                 attention_mode="dot", start_token="_START_", 
                 stop_token="_STOP_"):

        super(RNNDecoder, self).__init__()

        if hidden_dim is None:
            hidden_dim = embedding_dim
        self.hidden_dim_ = hidden_dim
        self.dropout_ = dropout
        self.num_layers_ = num_layers

        self.vocab_ = vocab        
        self.start_index_ = self.vocab.index(start_token)
        self.stop_index_ = self.vocab.index(stop_token)
        
        self.embeddings_ = nn.Embedding(
            vocab.size, embedding_dim, padding_idx=0)
        if rnn_type == "rnn":
            self.rnn_ = nn.RNN(
                embedding_dim, hidden_dim, nonlinearity="relu",
                num_layers=self.num_layers_,
                bidirectional=False)
        elif rnn_type == "lstm":
            self.rnn_ = nn.LSTM(
                embedding_dim, hidden_dim,
                num_layers=self.num_layers_,
                bidirectional=False)

        elif rnn_type == "gru":
            self.rnn_ = nn.GRU(
                embedding_dim, hidden_dim,
                num_layers=self.num_layers_,
                bidirectional=False)
        else:
            raise Exception("rnn_type {} not supported".format(rnn_type))
 


        if attention_mode == "dot":
            self.attention_module_ = attn.DotAttention(
                context_dropout=self.dropout_)
        elif attention_mode == "none":
            self.attention_module_ = None
        else:
            raise Exception("Attention mode {} not supported.".format(
                attention_mode))

        logits_dim = hidden_dim * 2 if self.has_attention else hidden_dim

        self.logits_layer_ = nn.Linear(logits_dim, self.vocab.size)

    @property
    def vocab(self):
        return self.vocab_

    @property
    def start_index(self):
        return self.start_index_
    
    @property
    def stop_index(self):
        return self.stop_index_

    @property
    def has_attention(self):
        return self.attention is not None

    @property
    def attention(self):
        return self.attention_module_

    def forward(self, prev_state, target_in, target_length, context=None):

        max_steps = target_in.size(1)
        batch_size = target_in.size(0)
        hidden_dim = self.hidden_dim_ 
        if self.has_attention:
            hidden_dim *= 2

        emb = self.embeddings_(target_in.t())

        dec_hidden, _ = self.rnn_(emb, prev_state)

        dec_hidden = F.dropout(
            dec_hidden, p=self.dropout_, training=self.training, inplace=False)

        if self.has_attention:
            weights, attn_context = self.attention(context, dec_hidden)
            hidden = torch.cat([dec_hidden, attn_context], 2)
            
        else:
            hidden = dec_hidden

        hidden_flat = hidden.view(
            max_steps * batch_size, hidden_dim)

        logits_flat = self.logits_layer_(hidden_flat)
        logits = logits_flat.view(max_steps, batch_size, self.vocab.size)

#        for b in range(batch_size):
#            len_b = target_length[b]
#            if len_b < max_steps:
#                logits.data[len_b:,b].fill_(0)
        return logits
    
    def greedy_predict(self, prev_state, context=None, max_steps=50):

        batch_size = prev_state.size(1)

        target_in = torch.LongTensor()
        target_in.resize_(1, batch_size).fill_(self.start_index)
        target_in = Variable(target_in)

        target_out = torch.LongTensor()
        target_out.resize_(batch_size, max_steps).fill_(0)

        not_stopped = torch.ByteTensor().resize_(batch_size).fill_(1)

        for i in range(max_steps):
            emb = self.embeddings_(target_in)
            

            dec_hidden = self.rnn_(emb, prev_state)[1]
            #print(dec_hidden)
            if self.has_attention:
                weights, attn_context = self.attention(context, dec_hidden)
                hidden = torch.cat([dec_hidden, attn_context], 2).squeeze(0)
            else:
                hidden = dec_hidden.squeeze(0)
            
            prev_state = dec_hidden

            logits = self.logits_layer_(hidden)
            max_val, pred_step = logits.max(1)
            target_in = pred_step.unsqueeze(0)

            target_out[:,i].copy_(pred_step.data)
            target_out[:,i].masked_fill_(~not_stopped, 0)
            
            not_stopped.mul_(pred_step.data.ne(self.stop_index))
            
            # I can't believe I have to go to numpy for this
            if ~not_stopped.numpy().any():
                target_out = target_out[:,:i+1]
                break
            
        
            
        return target_out


class RNNLTDecoder(nn.Module):

    def __init__(self, vocab, vocab_len, embedding_dim, 
                 len_dim,
                 rnn_type="gru",
                 num_layers=1,
                 hidden_dim=None, dropout=0.0,
                 attention_mode="dot", start_token="_START_", 
                 stop_token="_STOP_"):

        super(RNNLTDecoder, self).__init__()

        if hidden_dim is None:
            hidden_dim = embedding_dim
        self.hidden_dim_ = hidden_dim
        self.dropout_ = dropout
        self.num_layers_ = num_layers

        self.vocab_ = vocab      
        self.vocab_len_ = vocab_len
        self.start_index_ = self.vocab.index(start_token)
        self.stop_index_ = self.vocab.index(stop_token)
        
        self.embeddings_ = nn.Embedding(
            vocab.size, embedding_dim, padding_idx=0)
        self.embeddings_len_ = nn.Embedding(
            vocab_len.size, len_dim, padding_idx=0)

        if rnn_type == "rnn":
            self.rnn_ = nn.RNN(
                embedding_dim + len_dim, hidden_dim, nonlinearity="relu",
                num_layers=self.num_layers_,
                bidirectional=False)
        elif rnn_type == "lstm":
            self.rnn_ = nn.LSTM(
                embedding_dim + len_dim, hidden_dim,
                num_layers=self.num_layers_,
                bidirectional=False)

        elif rnn_type == "gru":
            self.rnn_ = nn.GRU(
                embedding_dim + len_dim, hidden_dim,
                num_layers=self.num_layers_,
                bidirectional=False)
        else:
            raise Exception("rnn_type {} not supported".format(rnn_type))
 
        if attention_mode == "dot":
            self.attention_module_ = attn.DotAttention(
                context_dropout=self.dropout_)
        elif attention_mode == "none":
            self.attention_module_ = None
        else:
            raise Exception("Attention mode {} not supported.".format(
                attention_mode))

        logits_dim = hidden_dim * 2 if self.has_attention else hidden_dim

        self.logits_layer_ = nn.Linear(logits_dim, self.vocab.size)

    @property
    def vocab(self):
        return self.vocab_

    @property
    def start_index(self):
        return self.start_index_
    
    @property
    def stop_index(self):
        return self.stop_index_

    @property
    def has_attention(self):
        return self.attention is not None

    @property
    def attention(self):
        return self.attention_module_

    def forward(self, prev_state, target_in, target_in_lt, target_length, 
                context=None):

        max_steps = target_in.size(1)
        batch_size = target_in.size(0)
        hidden_dim = self.hidden_dim_ 
        if self.has_attention:
            hidden_dim *= 2

        emb = self.embeddings_(target_in.t())

        len_emb = self.embeddings_len_(target_in_lt)
        len_emb = len_emb.expand(len_emb.size(0), emb.size(0), len_emb.size(2))
        len_emb = len_emb.permute(1, 0, 2)

        emb = torch.cat([emb, len_emb], 2)

        dec_hidden, _ = self.rnn_(emb, prev_state)

        dec_hidden = F.dropout(
            dec_hidden, p=self.dropout_, training=self.training, inplace=False)

        if self.has_attention:
            weights, attn_context = self.attention(context, dec_hidden)
            hidden = torch.cat([dec_hidden, attn_context], 2)
            
        else:
            hidden = dec_hidden

        hidden_flat = hidden.view(
            max_steps * batch_size, hidden_dim)

        logits_flat = self.logits_layer_(hidden_flat)
        logits = logits_flat.view(max_steps, batch_size, self.vocab.size)

#        for b in range(batch_size):
#            len_b = target_length[b]
#            if len_b < max_steps:
#                logits.data[len_b:,b].fill_(0)
        return logits
 

