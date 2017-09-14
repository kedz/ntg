import torch.nn as nn

class RNNEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim=None):

        super(RNNEncoder, self).__init__()

        if hidden_dim is None:
            hidden_dim = embedding_dim

        self.embeddings_ = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.rnn_ = nn.RNN(embedding_dim, hidden_dim)

    def get_output_state(self, encoder_output):
        return encoder_output[1]

    def get_output_context(self, encoder_output):
        return encoder_output[0]

    def forward(self, source, source_length):

        emb = self.embeddings_(source.t())
        emb_pkd = nn.utils.rnn.pack_padded_sequence(
            emb, source_length.tolist(), batch_first=False)
        return self.rnn_(emb_pkd)
        
        print(emb_pkd)
        print("")



        seq_src_pkd, hidden = self.rnn_(emb_pkd)

        seq_src, _ = nn.utils.rnn.pad_packed_sequence(
            seq_src_pkd, batch_first=True)

        print(seq_src)

        return hidden

        exit()

