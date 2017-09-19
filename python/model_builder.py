import encoder
import decoder
import base_model

def seq2seq(vocab_src, vocab_tgt, rnn_type="gru", embedding_dim=300, 
        hidden_dim=300, attention_type="dot", dropout=.5, num_layers=1,
        bidirectional=True):

    enc = encoder.RNNEncoder(
        vocab_src, embedding_dim, hidden_dim=hidden_dim,
        rnn_type=rnn_type,
        num_layers=num_layers,
        bidirectional=bidirectional)
    dec = decoder.RNNDecoder(
        vocab_tgt, embedding_dim, rnn_type=rnn_type,
        hidden_dim=hidden_dim,
        attention_mode=attention_type, dropout=dropout,
        num_layers=num_layers)

    return base_model.Seq2SeqModel(enc, dec)

def seq2seq_lt(vocab_src, vocab_tgt, vocab_len, rnn_type="rnn",
        num_layers=1,
        bidirectional=True, 
        embedding_dim=300, len_embedding_dim=50, 
        hidden_dim=300, attention_type="dot", dropout=.5):

    enc = encoder.RNNEncoder(
        vocab_src, embedding_dim, hidden_dim=hidden_dim,
        rnn_type=rnn_type,
        num_layers=num_layers,
        bidirectional=bidirectional)

    dec = decoder.RNNLTDecoder(
        vocab_tgt, vocab_len, embedding_dim, len_embedding_dim, 
        rnn_type=rnn_type, num_layers=num_layers,
        hidden_dim=hidden_dim,
        attention_mode=attention_type, dropout=dropout)

    return base_model.Seq2SeqLTModel(enc, dec)
