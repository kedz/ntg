from decoder.decoder_base import DecoderBase
import attention
import torch
import torch.nn as nn

class RNNRSDecoder(DecoderBase):
    def __init__(self, rnn_type, vocab_size, embedding_size, 
                 max_steps, step_embedding_size,
                 hidden_size, 
                 num_layers, attention_type="dot"):
        super(RNNRSDecoder, self).__init__()

        self.rnn_type_ = rnn_type
        self.vocab_size_ = vocab_size
        self.embedding_size_ = embedding_size
        self.hidden_size_ = hidden_size
        self.num_layers_ = num_layers
        self.attention_type_ = attention_type

        self.embeddings_ = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=0)
        self.step_embeddings_ = nn.Embedding(
            max_steps + 1, step_embedding_size, padding_idx=0)

        rnn_args = {"num_layers": num_layers, "bidirectional": False}
        if rnn_type == "rnn":
            rnn_init = nn.RNN
            rnn_args["nonlinearity"] = "relu"
        elif rnn_type == "gru":
            rnn_init = nn.GRU
        elif rnn_type == "lstm":
            rnn_init = nn.LSTM 
        else:
            raise Exception("rnn_type {} not supported".format(rnn_type))

        self.rnn_ = rnn_init(
            embedding_size + step_embedding_size, hidden_size, **rnn_args)

        if self.attention_type == "none":
            self.mlp_input_size_ = self.hidden_size
            def no_op(input, context=None):
                return input
            self.attention_ = no_op

        elif self.attention_type == "dot":
            self.attention_ = attention.DotAttention(merge_type="concat")
            self.mlp_input_size_ = self.hidden_size * 2
        else:
            raise Exception(
                "attention_type {} not implemented.".format(
                    self.attention_type))
        
        self.mlp_ = nn.Linear(self.mlp_input_size_, vocab_size)

    @property
    def num_layers(self):
        return self.num_layers_

    @property
    def rnn_type(self):
        return self.rnn_type_

    @property
    def attention_type(self):
        return self.attention_type_

    @property
    def attention(self):
        return self.attention_

    @property
    def hidden_size(self):
        return self.hidden_size_

    @property
    def embedding_size(self):
        return self.embedding_size_

    @property
    def vocab_size(self):
        return self.vocab_size_

    def forward(self, prev_state, input, steps, context=None): 
        
        max_steps = input.size(1)
        batch_size = input.size(0)

        word_embeddings = self.embeddings_(input.t())
        rs_embeddings = self.step_embeddings_(steps.t())

        embeddings = torch.cat([word_embeddings, rs_embeddings], 2)
        rnn_output, rnn_state = self.rnn_(embeddings, prev_state)

        rnn_output = self.attention(rnn_output, context=context)

        rnn_output_flat = rnn_output.view(
            max_steps * batch_size, self.mlp_input_size_)

        logits = self.mlp_(rnn_output_flat).view(
            max_steps, batch_size, self.vocab_size)

        logits_mask = input.data.t().eq(0).view(
            max_steps, batch_size, 1).expand(
            max_steps, batch_size, self.vocab_size)
        logits.data.masked_fill_(logits_mask, 0)

        return logits
        
    
    def greedy_predict(self, batch):
        pass
