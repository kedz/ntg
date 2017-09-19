import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNEncoder(nn.Module):

    def __init__(self, vocab, embedding_dim, rnn_type="gru", hidden_dim=None,
                 bidirectional=True,
                 num_layers=3,
                 bridge="linear-relu"):

        super(RNNEncoder, self).__init__()

        if hidden_dim is None:
            hidden_dim = embedding_dim

        self.embedding_dim_ = embedding_dim
        self.bidirectional_ = bidirectional
        self.num_layers_ = num_layers
        self.rnn_type_ = rnn_type

        self.vocab_ = vocab

        self.embeddings_ = nn.Embedding(
            vocab.size, embedding_dim, padding_idx=0)

        if rnn_type == "rnn":
            self.rnn_ = nn.RNN(
                embedding_dim, hidden_dim, nonlinearity="relu",
                num_layers=self.num_layers_,
                bidirectional=self.bidirectional_)
        elif rnn_type == "lstm":
            self.rnn_ = nn.LSTM(
                embedding_dim, hidden_dim,
                num_layers=self.num_layers_,
                bidirectional=self.bidirectional_)

        elif rnn_type == "gru":
            self.rnn_ = nn.GRU(
                embedding_dim, hidden_dim,
                num_layers=self.num_layers_,
                bidirectional=self.bidirectional_)
        else:
            raise Exception("rnn_type {} not supported".format(rnn_type))
            
        if self.bidirectional_:

            if bridge == "average":
                self.bridge = self.bridge_average_
            elif bridge == "linear-relu":
                self.bridge_linear_layer_ = nn.Linear(
                    hidden_dim * 2, hidden_dim)
                if rnn_type == "lstm":
                    self.bridge_linear_layer_cell_ = nn.Linear(
                        hidden_dim * 2, hidden_dim)
                
                self.bridge = self.bridge_linear_relu_

            else:
                raise Exception("bridge {} not supported".format(bridge))



    def bridge_average_(self, state):
        if self.rnn_type_ != "lstm":
            state_expand = state.view(
                self.num_layers_, 2, state.size(1), state.size(2))
            fwd_state = state_expand[:,0]
            bwd_state = state_expand[:,1]
           
            bridge_state = (fwd_state + bwd_state) / 2
            return bridge_state
        else: 
            state_expand_0 = state[0].view(
                self.num_layers_, 2, state[0].size(1), state[0].size(2))
            fwd_state_0 = state_expand_0[:,0]
            bwd_state_0 = state_expand_0[:,1]
           
            bridge_state_0 = (fwd_state_0 + bwd_state_0) / 2
 
            state_expand_1 = state[1].view(
                self.num_layers_, 2, state[1].size(1), state[1].size(2))
            fwd_state_1 = state_expand_1[:,0]
            bwd_state_1 = state_expand_1[:,1]
           
            bridge_state_1 = (fwd_state_1 + bwd_state_1) / 2
            return bridge_state_0, bridge_state_1


    def bridge_linear_relu_(self, state):

        if self.rnn_type_ != "lstm":
            state_expand = state.view(
                self.num_layers_, 2, state.size(1), state.size(2))
            fwd_state = state_expand[:,0]
            bwd_state = state_expand[:,1]

            state_cat = torch.cat([fwd_state, bwd_state], 2)

            state_cat_flat = state_cat.view(
                self.num_layers_ * state.size(1), state.size(2) * 2)

            bridge_state_flat = F.relu(
                self.bridge_linear_layer_(state_cat_flat))
            bridge_state = bridge_state_flat.view(
                self.num_layers_, state.size(1), state.size(2))
            return bridge_state
        else:
            state_expand_h = state[0].view(
                self.num_layers_, 2, state[0].size(1), state[0].size(2))
            fwd_state_h = state_expand_h[:,0]
            bwd_state_h = state_expand_h[:,1]

            state_cat_h = torch.cat([fwd_state_h, bwd_state_h], 2)

            state_cat_flat_h = state_cat_h.view(
                self.num_layers_ * state[0].size(1), state[0].size(2) * 2)

            bridge_state_flat_h = F.relu(
                self.bridge_linear_layer_(state_cat_flat_h))
            bridge_state_h = bridge_state_flat_h.view(
                self.num_layers_, state[0].size(1), state[0].size(2))


            state_expand_c = state[1].view(
                self.num_layers_, 2, state[1].size(1), state[1].size(2))
            fwd_state_c = state_expand_c[:,0]
            bwd_state_c = state_expand_c[:,1]

            state_cat_c = torch.cat([fwd_state_c, bwd_state_c], 2)

            state_cat_flat_c = state_cat_c.view(
                self.num_layers_ * state[1].size(1), state[1].size(2) * 2)

            bridge_state_flat_c = F.relu(
                self.bridge_linear_layer_cell_(state_cat_flat_c))
            bridge_state_c = bridge_state_flat_c.view(
                self.num_layers_, state[1].size(1), state[1].size(2))

            return bridge_state_h, bridge_state_c

 
    def get_output_state(self, encoder_output):
        return encoder_output[1]

    def get_output_context(self, encoder_output):
        return encoder_output[0]

    def forward(self, source, source_length):

        emb = self.embeddings_(source.t())
        emb_pkd = nn.utils.rnn.pack_padded_sequence(
            emb, source_length.tolist(), batch_first=False)

        state = self.rnn_(emb_pkd)

        if self.bidirectional_:
            output = state[0]
            output, lens = nn.utils.rnn.pad_packed_sequence(output)
            output = output.view(output.size(0), output.size(1), 
                2, self.rnn_.hidden_size).sum(2)
            output = nn.utils.rnn.pack_padded_sequence(
                output, lens, batch_first=False)

            h_t = self.bridge(state[1])

            state = (output, h_t)
                 
        return state

