from models.sequence_model_base import SequenceModelBase
import torch
import torch.nn as nn


class Seq2SeqRS(SequenceModelBase):

    def __init__(self, encoder, bridge, decoder):

        super(Seq2SeqRS, self).__init__()
        self.encoder_ = encoder
        self.bridge_ = bridge
        self.decoder_ = decoder

    @property
    def encoder(self):
        return self.encoder_
    
    @property
    def bridge(self):
        return self.bridge_

    @property
    def decoder(self):
        return self.decoder_

    def forward(self, batch):
        print("")
        #print(batch.encoder_inputs) 
        print(batch.target) 
        encoder_output, encoder_state = self.encoder(
            batch.encoder_inputs, batch.encoder_input_length)

        decoder_state = self.bridge(encoder_state)

        logits = self.decoder(
            batch.decoder_inputs, 
            init_state=decoder_state,
            context=encoder_output)
        
        return logits

    def greedy_predict(self, args):
        pass

#    def __init__(self, rnn_type, encoder_vocab, decoder_vocab, 
#                 embedding_size, hidden_size, num_layers, 
#                 bidirectional,
#                 bridge_type="linear-relu", attention_type="dot",
#                 max_steps=50, step_embedding_size=100,
#                 learn_start=True):
#        super(Seq2SeqRS, self).__init__()
#
#        self.rnn_type_ = rnn_type
#        self.encoder_vocab_ = encoder_vocab
#        self.decoder_vocab_ = decoder_vocab
#        self.embedding_size_ = embedding_size
#        self.hidden_size_ = hidden_size
#        self.num_l/forayers_ = num_layers
#        self.bidirectional_ = bidirectional
#        self.attention_type_ = attention_type
#
#        self.encoder_ = RNNEncoder(
#            self.rnn_type,
#            self.encoder_vocab.size,
#            self.embedding_size,
#            self.hidden_size,
#            num_layers=self.num_layers,
#            bidirectional=self.bidirectional_encoder)
#
#        if learn_start:
#            state_size = self.num_layers * (2 if bidirectional else 1)
#            self.init_hidden = nn.Parameter(
#                torch.FloatTensor(state_size, 1, self.hidden_size))
#            nn.init.normal(self.init_hidden)
#
#            if self.rnn_type == "lstm":
#                self.init_cell = nn.Parameter(
#                    torch.FloatTensor(state_size, 1, self.hidden_size))
#                nn.init.normal(self.init_cell)
#            else:
#                self.register_parameter('init_cell', None)
#        else:
#            self.register_parameter('init_hidden', None)
#            self.register_parameter('init_cell', None)
#
#        if self.bidirectional_encoder:
#            if bridge_type.startswith("linear-"):
#                act_type = bridge_type.split("-")[1]
#                self.bridge_ = bridge.FullyConnectedBridge(
#                    self.hidden_size, self.num_layers, 
#                    activation=act_type, bidirectional_input=True)
#            elif bridge_type == "average":
#                self.bridge_ = bridge.AveragingBridge(self.num_layers)
#            else:
#                raise Exception(
#                    "bridge_type {} not valid.".format(bridge_type))
#
#        else:
#            self.bridge_ = bridge.NoOpBridge()
#
#        self.decoder_ = RNNRSDecoder(
#            self.rnn_type,
#            self.decoder_vocab.size,
#            self.embedding_size,
#            max_steps,
#            step_embedding_size,
#            self.hidden_size,
#            self.num_layers,
#            attention_type=self.attention_type)
#
#    @property
#    def encoder(self):
#        return self.encoder_
#
#    @property
#    def bridge(self):
#        return self.bridge_
#
#    @property
#    def decoder(self):
#        return self.decoder_
#
#    @property
#    def embedding_size(self):
#        return self.embedding_size_
#
#    @property
#    def hidden_size(self):
#        return self.hidden_size_
#
#    @property
#    def num_layers(self):
#        return self.num_layers_
#
#    @property
#    def rnn_type(self):
#        return self.rnn_type_
#
#    @property
#    def attention_type(self):
#        return self.attention_type_
#
#    @property
#    def encoder_vocab(self):
#        return self.encoder_vocab_
#
#    @property
#    def decoder_vocab(self):
#        return self.decoder_vocab_
#
#    @property
#    def bidirectional_encoder(self):
#        return self.bidirectional_
#
#
#    def init_state(self, batch_size):
#        if self.init_hidden is None:
#            return None
#
#        init_hidden = self.init_hidden.repeat(1, batch_size, 1)
#
#        if self.init_cell is None:
#            return init_hidden
#        else:
#            init_cell = self.init_cell.repeat(1, batch_size, 1)
#            return (init_hidden, init_cell)
#
#    def encode_batch(self, batch):
#        
#        # TODO: do init cell stuff here
#        encoder_output, encoder_state = self.encoder(
#            batch.encoder_input, batch.encoder_input_length)
#        return encoder_output, encoder_state        
# 
#    def forward(self, batch):
#
#        encoder_output, encoder_state = self.encoder(
#            batch.encoder_input, batch.encoder_input_length)
#
#        decoder_state = self.bridge(encoder_state)
#
#        logits = self.decoder(
#            decoder_state, batch.decoder_input, batch.remaining_steps,
#            context=encoder_output)
#
#        return logits
#
#    def greedy_predict(self, batch, return_logits=False, max_steps=100):
#        pass
