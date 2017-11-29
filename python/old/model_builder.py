from input_module import DiscreteFeatureSequenceInput
from encoder import RNNEncoder
from decoder import RNNDecoder
import attention 
import bridge
from models import Seq2SeqRNN, Seq2SeqRNNRS, Seq2SeqRNNTS
import torch.nn as nn

class Seq2SeqRNNBuilder(object):

    def __init__(self):

        self.encoder = None
        self.bridge = None
        self.decoder = None

    def add_encoder(self, vocab_sizes, embedding_sizes, rnn_type, hidden_size,
                    num_layers, bidirectional):
        encoder_input = DiscreteFeatureSequenceInput(
            vocab_sizes, embedding_sizes)
        encoder = RNNEncoder(
            encoder_input, rnn_type, hidden_size, num_layers, bidirectional)
        self.encoder = encoder
        return self   

    def add_decoder(self, vocab_sizes, embedding_sizes, rnn_type, hidden_size,
                    num_layers, output_vocab_size, start_index, stop_index,
                    attention_type):

        decoder_input = DiscreteFeatureSequenceInput(
            vocab_sizes, embedding_sizes)

        rnn_args = {"num_layers": num_layers, "bidirectional": False}
        if rnn_type == "rnn":
            rnn_init = nn.RNN
            rnn_args["nonlinearity"] = "relu"
        elif rnn_type == "gru":
            rnn_init = nn.GRU
        elif rnn_type == "lstm":
            rnn_init = nn.LSTM 
        else:
            raise Exception("rnn_type {} not supported".format(self.rnn_type))

        rnn_module = rnn_init(
            decoder_input.embedding_size, hidden_size, **rnn_args)
 
        mlp_input_size = hidden_size
        
        if attention_type == "none":
            attention_module = attention.NoOpAttention()
        elif attention_type == "dot":
            attention_module = attention.DotAttention(merge_type="concat")
            mlp_input_size *= 2
        else:
            raise Exception(
                "attention_type {} not implemented.".format(
                    self.attention_type))
        predictor_module = nn.Linear(mlp_input_size, output_vocab_size)

        decoder = RNNDecoder(
            decoder_input, rnn_module, attention_module, predictor_module,
            start_index, stop_index)

        self.decoder = decoder

        return self

    def add_bridge(self, bridge_type, bidirectional_encoder, hidden_size, 
                   num_layers):

        if bidirectional_encoder:
            if bridge_type.startswith("linear-"):
                act_type = bridge_type.split("-")[1]
                self.bridge = bridge.FullyConnectedBridge(
                    hidden_size, num_layers, 
                    activation=act_type, bidirectional_input=True)
            elif bridge_type == "average":
                self.bridge = bridge.AveragingBridge(self.num_layers)
            else:
                raise Exception(
                    "bridge_type {} not valid.".format(bridge_type))

        else:
            self.bridge = bridge.NoOpBridge()

    def finish_rnn(self):
        return Seq2SeqRNN(self.encoder, self.bridge, self.decoder)
    def finish_rnn_rs(self):
        return Seq2SeqRNNRS(self.encoder, self.bridge, self.decoder)
    def finish_rnn_ts(self):
        return Seq2SeqRNNTS(self.encoder, self.bridge, self.decoder)

#import encoder
#import decoder
#import base_model
#
#def seq2seq(vocab_src, vocab_tgt, rnn_type="gru", embedding_dim=300, 
#        hidden_dim=300, attention_type="dot", dropout=.5, num_layers=1,
#        bidirectional=True):
#
#    enc = encoder.RNNEncoder(
#        vocab_src, embedding_dim, hidden_dim=hidden_dim,
#        rnn_type=rnn_type,
#        num_layers=num_layers,
#        bidirectional=bidirectional)
#    dec = decoder.RNNDecoder(
#        vocab_tgt, embedding_dim, rnn_type=rnn_type,
#        hidden_dim=hidden_dim,
#        attention_mode=attention_type, dropout=dropout,
#        num_layers=num_layers)
#
#    return base_model.Seq2SeqModel(enc, dec)
#
#def seq2seq_lt(vocab_src, vocab_tgt, vocab_len, rnn_type="rnn",
#        num_layers=1,
#        bidirectional=True, 
#        embedding_dim=300, len_embedding_dim=50, 
#        hidden_dim=300, attention_type="dot", dropout=.5):
#
#    enc = encoder.RNNEncoder(
#        vocab_src, embedding_dim, hidden_dim=hidden_dim,
#        rnn_type=rnn_type,
#        num_layers=num_layers,
#        bidirectional=bidirectional)
#
#    dec = decoder.RNNLTDecoder(
#        vocab_tgt, vocab_len, embedding_dim, len_embedding_dim, 
#        rnn_type=rnn_type, num_layers=num_layers,
#        hidden_dim=hidden_dim,
#        attention_mode=attention_type, dropout=dropout)
#
#    return base_model.Seq2SeqLTModel(enc, dec)
#
#
#def seq2seq_sp1(vocab_src, vocab_tgt, rnn_type="gru", embedding_dim=300, 
#        hidden_dim=300, attention_type="dot", dropout=.5, num_layers=1,
#        bidirectional=True):
#
#    enc = encoder.RNNEncoder(
#        vocab_src, embedding_dim, hidden_dim=hidden_dim,
#        rnn_type=rnn_type,
#        num_layers=num_layers,
#        bidirectional=bidirectional)
#    dec = decoder.RNNDecoder(
#        vocab_tgt, embedding_dim, rnn_type=rnn_type,
#        hidden_dim=hidden_dim,
#        attention_mode=attention_type, dropout=dropout,
#        num_layers=num_layers)
#
#    return base_model.Seq2SeqSPModel(enc, dec)
#
#
