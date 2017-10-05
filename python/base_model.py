import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqModel(nn.Module):

    def __init__(self, encoder, decoder):

        super(Seq2SeqModel, self).__init__()
        
        self.encoder_ = encoder
        self.decoder_ = decoder

    @property
    def encoder(self):
        return self.encoder_

    @property
    def decoder(self):
        return self.decoder_

    def forward(self, batch):
        enc_out = self.encoder(batch.source, batch.source_length)
        enc_state = self.encoder.get_output_state(enc_out)

        if self.decoder.has_attention:
            context = self.encoder.get_output_context(enc_out)
        else:
            context = None

        logits = self.decoder(
            enc_state, batch.target_in, batch.target_length, context=context)

        return logits

    def predict(self, batch, teacher_forcing=False, beam_size=1, eval=True):

        if eval:
            self.eval()

        if teacher_forcing:
            prediction = self.predict_teach_forcing_(batch)

        elif beam_size == 1:
            prediction = self.predict_greedy_(
                batch.source, batch.source_length)

        else:
            print("Implement beam search")
            exit()

        return prediction

    def predict_teach_forcing_(self, batch):
        logits = self.forward(batch).data
        
        max_logits, pred_output = logits.max(2)
        pred_output = pred_output.t()
        
        max_steps = pred_output.size(1)
        for b in range(batch.target_length.size(0)):
            b_len = batch.target_length[b]
            if b_len < max_steps:
                pred_output[b,b_len:].fill_(0)

        return pred_output

        
    def predict_greedy_(self, source, source_length, max_steps=100):
        enc_out = self.encoder(source, source_length)
        enc_state = self.encoder.get_output_state(enc_out)

        if self.decoder.has_attention:
            context = self.encoder.get_output_context(enc_out)
        else:
            context = None

        return self.decoder.greedy_predict(enc_state, context=context)


class Seq2SeqSPModel(nn.Module):

    def __init__(self, encoder, decoder):

        super(Seq2SeqSPModel, self).__init__()
        
        self.encoder_ = encoder
        self.decoder_ = decoder

        self.topic_layer_ = nn.Linear(
            self.encoder_.rnn_.hidden_size, self.decoder.vocab_.size)

    @property
    def encoder(self):
        return self.encoder_

    @property
    def decoder(self):
        return self.decoder_

    def forward(self, batch):
        enc_out = self.encoder(batch.source, batch.source_length)

        rnn_output, lens = nn.utils.rnn.pad_packed_sequence(enc_out[0])

        o1 = rnn_output.sum(0)
        o2 = (o1 - o1.mean(1, keepdim=True)) / (o1.std(1, keepdim=True) + .000001)

        topics_logits = F.tanh(self.topic_layer_(o2))

        enc_state = self.encoder.get_output_state(enc_out)

        if self.decoder.has_attention:
            context = self.encoder.get_output_context(enc_out)
        else:
            context = None

        logits = self.decoder(
            enc_state, batch.target_in, batch.target_length, context=context)

        return logits, topics_logits

    def predict(self, batch, teacher_forcing=False, beam_size=1, eval=True):

        if eval:
            self.eval()

        if teacher_forcing:
            prediction = self.predict_teach_forcing_(batch)

        elif beam_size == 1:
            prediction = self.predict_greedy_(
                batch.source, batch.source_length)

        else:
            print("Implement beam search")
            exit()

        return prediction

    def predict_teach_forcing_(self, batch):
        logits = self.forward(batch).data
        
        max_logits, pred_output = logits.max(2)
        pred_output = pred_output.t()
        
        max_steps = pred_output.size(1)
        for b in range(batch.target_length.size(0)):
            b_len = batch.target_length[b]
            if b_len < max_steps:
                pred_output[b,b_len:].fill_(0)

        return pred_output

        
    def predict_greedy_(self, source, source_length, max_steps=100):
        enc_out = self.encoder(source, source_length)
        enc_state = self.encoder.get_output_state(enc_out)

        if self.decoder.has_attention:
            context = self.encoder.get_output_context(enc_out)
        else:
            context = None

        return self.decoder.greedy_predict(enc_state, context=context)




class Seq2SeqLTModel(nn.Module):

    def __init__(self, encoder, decoder):

        super(Seq2SeqLTModel, self).__init__()
        
        self.encoder_ = encoder
        self.decoder_ = decoder

    @property
    def encoder(self):
        return self.encoder_

    @property
    def decoder(self):
        return self.decoder_

    def forward(self, batch):
        enc_out = self.encoder(batch.source, batch.source_length)
        enc_state = self.encoder.get_output_state(enc_out)

        if self.decoder.has_attention:
            context = self.encoder.get_output_context(enc_out)
        else:
            context = None

        logits = self.decoder(
            enc_state, batch.target_in, batch.target_len_tok_, 
            batch.target_length, context=context)

        return logits

    def predict(self, batch, teacher_forcing=False, beam_size=1, eval=True,
                test_zero=False, return_logits=False):

        if eval:
            self.eval()

        if teacher_forcing:
            prediction = self.predict_teach_forcing_(batch)

        elif beam_size == 1:
            prediction = self.predict_greedy_(
                batch.source, batch.source_length, batch.target_len_tok_,
                test_zero=test_zero, return_logits=return_logits)

        else:
            print("Implement beam search")
            exit()

        return prediction

    def predict_teach_forcing_(self, batch):
        logits = self.forward(batch).data
        
        max_logits, pred_output = logits.max(2)
        pred_output = pred_output.t()
        
        max_steps = pred_output.size(1)
        for b in range(batch.target_length.size(0)):
            b_len = batch.target_length[b]
            if b_len < max_steps:
                pred_output[b,b_len:].fill_(0)

        return pred_output

        
    def predict_greedy_(self, source, source_length, len_tok, max_steps=100,
            test_zero=False, return_logits=False):
        enc_out = self.encoder(source, source_length)
        enc_state = self.encoder.get_output_state(enc_out)

        if self.decoder.has_attention:
            context = self.encoder.get_output_context(enc_out)
        else:
            context = None

        pred, logits = self.decoder.greedy_predict(
            enc_state, len_tok, context=context, max_steps=max_steps,
            test_zero=test_zero)
        if return_logits:
            return pred, logits
        else:
            return pred



