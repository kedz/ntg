from models.sequence_model_base import SequenceModelBase
from decoder import RNNLMDecoder
import torch
import torch.nn as nn

class RNNLM(SequenceModelBase):

    def __init__(self, rnn_type, vocab, embedding_size, hidden_size, 
                 num_layers, learn_start=True):
        super(RNNLM, self).__init__()
        self.rnn_type_ = rnn_type
        self.vocab_ = vocab
        self.embedding_size_ = embedding_size
        self.hidden_size_ = hidden_size
        self.num_layers_ = num_layers
        self.learn_start_ = learn_start

        self.decoder_ = RNNLMDecoder(
            self.rnn_type,
            self.vocab.size, 
            self.embedding_size,
            self.hidden_size, 
            self.num_layers,
            attention_type="none")

        if learn_start:
            self.start_hidden_ = nn.Parameter(
                torch.FloatTensor(num_layers, 1, hidden_size))
            nn.init.normal(self.start_hidden_)

            if rnn_type == "lstm":
                self.start_cell_ = nn.Parameter(
                    torch.FloatTensor(num_layers, 1, hidden_size))
                nn.init.normal(self.start_cell_)

        else:
            self.start_hidden_ = Variable(
                torch.FloatTensor(num_layers, 1, hidden_size).fill_(0))
            if rnn_type == "lstm":
                self.start_cell_ = Variable(
                    torch.FloatTensor(num_layers, 1, hidden_size).fill_(0))

        

    @property
    def decoder(self):
        return self.decoder_

    @property
    def vocab(self):
        return self.vocab_

    @property
    def embedding_size(self):
        return self.embedding_size_

    @property
    def hidden_size(self):
        return self.hidden_size_

    @property
    def num_layers(self):
        return self.num_layers_

    @property
    def rnn_type(self):
        return self.rnn_type_

    def get_start_state(self, batch_size):
        hidden_state = self.start_hidden_.repeat(1, batch_size, 1)
        if self.rnn_type == "lstm":
            cell_state = self.start_cell_.repeat(1, batch_size, 1)
            return hidden_state, cell_state
        else:
            return hidden_state

    def forward(self, batch):

        hidden_state = self.get_start_state(batch.input.size(0))
        return self.decoder(hidden_state, batch.input) 


    def greedy_predict(self, batch, return_logits=False, max_steps=100):
        pass

   
