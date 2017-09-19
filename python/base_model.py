import torch.nn as nn


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

#    def predict(self, batch, teacher_forcing=False, beam_size=1, eval=True):
#
#        if eval:
#            self.eval()
#
#        if teacher_forcing:
#            prediction = self.predict_teach_forcing_(batch)
#
#        elif beam_size == 1:
#            prediction = self.predict_greedy_(
#                batch.source, batch.source_length)
#
#        else:
#            print("Implement beam search")
#            exit()
#
#        return prediction
#
#    def predict_teach_forcing_(self, batch):
#        logits = self.forward(batch).data
#        
#        max_logits, pred_output = logits.max(2)
#        pred_output = pred_output.t()
#        
#        max_steps = pred_output.size(1)
#        for b in range(batch.target_length.size(0)):
#            b_len = batch.target_length[b]
#            if b_len < max_steps:
#                pred_output[b,b_len:].fill_(0)
#
#        return pred_output
#
#        
#    def predict_greedy_(self, source, source_length, max_steps=100):
#        enc_out = self.encoder(source, source_length)
#        enc_state = self.encoder.get_output_state(enc_out)
#
#        if self.decoder.has_attention:
#            context = self.encoder.get_output_context(enc_out)
#        else:
#            context = None
#
#        return self.decoder.greedy_predict(enc_state, context=context)
#
#
#

