import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import namedtuple
from decoder.decoder_base import DecoderBase
import attention

class RNNLMDecoder(DecoderBase):
    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, 
                 num_layers, attention_type="dot"):
        super(RNNLMDecoder, self).__init__()

        self.rnn_type_ = rnn_type
        self.vocab_size_ = vocab_size
        self.embedding_size_ = embedding_size
        self.hidden_size_ = hidden_size
        self.num_layers_ = num_layers
        self.attention_type_ = attention_type

        self.embeddings_ = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=0)
        
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

        self.rnn_ = rnn_init(embedding_size, hidden_size, **rnn_args)

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

    def forward(self, prev_state, input, context=None): 
        
        max_steps = input.size(1)
        batch_size = input.size(0)

        embeddings = self.embeddings_(input.t())
        rnn_output, rnn_state = self.rnn_(embeddings, prev_state)

        rnn_output, _ = self.attention(rnn_output, context=context)

        rnn_output_flat = rnn_output.view(
            max_steps * batch_size, self.mlp_input_size_)

        logits = self.mlp_(rnn_output_flat).view(
            max_steps, batch_size, self.vocab_size)

        logits_mask = input.data.t().eq(0).view(
            max_steps, batch_size, 1).expand(
            max_steps, batch_size, self.vocab_size)
        logits.data.masked_fill_(logits_mask, 0)

        return logits
        
    
    def greedy_predict(self, prev_state, start_index, stop_index, 
                       context=None, return_attention=True, max_steps=50):

        batch_size = prev_state.size(1)

        decoder_input = prev_state.data.new().long()
        decoder_input.resize_(1, batch_size).fill_(start_index) 
        decoder_input = Variable(decoder_input)

        target = decoder_input.data.new()
        target.resize_(batch_size, max_steps).fill_(0)

        not_stopped = decoder_input.data.new().byte().resize_(batch_size)
        not_stopped.fill_(1)

        if return_attention:
            attention_steps = []

        for i in range(max_steps):
            
            embeddings = self.embeddings_(decoder_input)
            rnn_output, rnn_state = self.rnn_(embeddings, prev_state)
            mlp_input, attention_step = self.attention(
                rnn_output, context=context)
            logits = self.mlp_(mlp_input[0])
            if return_attention:
                attention_steps.append(attention_step.data)

            max_val, pred_step = logits.max(1)
            decoder_input = Variable(pred_step.unsqueeze(0).data)
            prev_state = rnn_state

            target[:,i].copy_(pred_step.data)
            target[:,i].masked_fill_(~not_stopped, 0)

            output_non_stop_index = pred_step.data.ne(stop_index)
            not_stopped.mul_(output_non_stop_index)
            if not_stopped.sum() == 0:
                break

        target_length = target.gt(0).sum(1)

        outputs = [target, target_length]
        output_names = ["target", "target_length"]


        if return_attention:
            attention = torch.cat(attention_steps, 1)
            outputs.append(attention)
            output_names.append("attention")
        DecoderOutput = namedtuple(
            "DecoderOutput", output_names) 

        return DecoderOutput(*outputs)
