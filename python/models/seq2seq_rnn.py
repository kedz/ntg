from models.sequence_model_base import SequenceModelBase
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqRNN(SequenceModelBase):

    def __init__(self, encoder, bridge, decoder):

        super(Seq2SeqRNN, self).__init__()
        self.encoder_ = encoder
        self.bridge_ = bridge
        self.decoder_ = decoder
        self.decoder_input_buffer_ = torch.LongTensor()

    def init_predict(self, batch_size, max_steps):
        self.decoder_input_buffer_ = torch.LongTensor()
        self.decoder_input_buffer_.resize_(batch_size, max_steps)
        self.decoder_input_buffer_[:,1:].fill_(0)
        self.decoder_input_buffer_[:,0].fill_(self.decoder.start_index)
        self.decoder_output_buffer_ = torch.LongTensor()
        self.decoder_output_buffer_.resize_(batch_size, max_steps)
        self.decoder_output_buffer_.fill_(0)
        return self.decoder_input_buffer_, self.decoder_output_buffer_

    def setup_inputs(self, decoder_input_buffer, step, extra_inputs=None):
        return [decoder_input_buffer[:,step:step + 1]]
    
    def get_next_inputs(self, prev_inputs, logits, rnn_state):
        val, max = logits.data.max(1)
        return max, rnn_state

    @property
    def encoder(self):
        return self.encoder_
    
    @property
    def bridge(self):
        return self.bridge_

    @property
    def decoder(self):
        return self.decoder_

    def encode(self, encoder_inputs, encoder_input_length):
        encoder_output, encoder_state = self.encoder(
            encoder_inputs, encoder_input_length)
        decoder_state = self.bridge(encoder_state)
        return encoder_output, decoder_state

    def forward(self, batch):
        encoder_output, decoder_state = self.encode(
            batch.encoder_inputs, batch.encoder_input_length)
        logits = self.decoder(
            batch.decoder_inputs, 
            init_state=decoder_state,
            context=encoder_output)        
        return logits



    def greedy_predict(self, encoder_inputs, encoder_input_length, 
                       extra_inputs=None, max_steps=100):
        batch_size = encoder_inputs[0].size(0)

        encoder_output, decoder_state = self.encode(
            encoder_inputs, encoder_input_length)
        decoder_input_buffer, decoder_output_buffer = self.init_predict(
            batch_size, max_steps)

        prev_state = decoder_state

        not_stopped = decoder_input_buffer.new().byte().resize_(batch_size)
        not_stopped.fill_(1)

        actual_steps = max_steps
        for step in range(max_steps-1):
            decoder_input_t = self.setup_inputs(
                decoder_input_buffer, step, extra_inputs=extra_inputs)
            logits, attention, rnn_state = self.decoder.forward_step(
                decoder_input_t, 
                prev_state=prev_state, 
                context=encoder_output)
            
            next_input, next_state = self.get_next_inputs(
                decoder_input_buffer[:,:step+1], logits, rnn_state)
            
            next_input.masked_fill_(~not_stopped, 0)

            decoder_output_buffer[:,step].copy_(next_input)
            decoder_input_buffer[:,step + 1].copy_(next_input)
            prev_state = next_state

            output_non_stop_index = next_input.ne(self.decoder.stop_index)
            not_stopped.mul_(output_non_stop_index)
            if not_stopped.sum() == 0:
                actual_steps = step + 1
                break

        return decoder_output_buffer[:,:actual_steps]



    def beam_step(self, logits, state, beam_scores):
        batch_size = beam_scores.size(0)
        beam_size = beam_scores.size(1)

        lsm = F.log_softmax(logits)
        lsm = lsm.view(batch_size, beam_size, logits.size(1))

        step_scores, indices = lsm.topk(beam_size, dim=2)
        #print(step_scores)
        #print(beam_scores)
        step_scores.add_(
            beam_scores.view(batch_size, beam_size, 1).repeat(
                1, 1, beam_size))
        #print(step_scores)

        next_state = state.data.new().resize_(
            state.size(0), batch_size * beam_size, state.size(2)).fill_(0)
        outputs = indices.data.new().resize_(1, batch_size, beam_size)
        back_pointers = indices.data.new().resize_(1, batch_size, beam_size)
        next_scores = step_scores.data.new().resize_(batch_size, beam_size)


        for batch in range(batch_size):
            predictions = []
            for beam_in in range(beam_size):
                for beam_out in range(beam_size):
                    score = step_scores.data[batch, beam_in, beam_out]
                    predictions.append((score, beam_in, beam_out))
            predictions.sort(key=lambda x: x[0], reverse=True)

            for beam_out, prediction in enumerate(predictions[:beam_size]):
                score, beam_in, loc = prediction
                next_state[:,batch * beam_size + beam_out,:].copy_(
                    state.data[:,batch * beam_size + beam_in,:])

                outputs[0,batch,beam_out] = indices.data[batch, beam_in, loc]
                back_pointers[0, batch, beam_out] = beam_in
                next_scores[batch, beam_out] = score

                    
        
        #print(indices)
        #print(outputs)
        #print(back_pointers)
        #print(next_scores)

        return outputs, Variable(next_state), Variable(next_scores), back_pointers



        exit()



    def beam_search(self, encoder_inputs, encoder_input_length, max_steps=250, 
                    beam_size=16, extra_inputs=None):

        batch_size = encoder_inputs[0].size(0)
        scores = Variable(torch.FloatTensor(batch_size, beam_size).fill_(0))
        scores.data[:,1:].fill_(float("-inf"))
        back_pointers = []


        encoder_output, decoder_state = self.encode(
            encoder_inputs, encoder_input_length)
        decoder_input_buffer, decoder_output_buffer = self.init_predict(
            batch_size * beam_size, max_steps)

        decoder_state = decoder_state.repeat(1, 1, beam_size).view(
            decoder_state.size(0), beam_size * batch_size, 
            decoder_state.size(2))
        encoder_output = encoder_output.repeat(1, 1, beam_size).view(
            encoder_output.size(0), beam_size * batch_size, 
            encoder_output.size(2))

        current_best = {b: [[], float("-inf"), []] for b in range(batch_size)}

        prev_state = decoder_state

        active_beams = [True for b in range(batch_size)]
        


        for step in range(max_steps - 1):
            #print("STEP: ", step)
            decoder_input_t = self.setup_inputs(
                decoder_input_buffer, step, extra_inputs=extra_inputs)
            #print(decoder_input_t)
            logits, attention, rnn_state = self.decoder.forward_step(
                decoder_input_t, 
                prev_state=prev_state, 
                context=encoder_output)
            #print(logits)
            #print(rnn_state)

            output, prev_state, scores, bp = self.beam_step(
                logits, rnn_state, scores)


            for batch in range(batch_size):
                if active_beams[batch]:
                    if scores.data[batch][0] < current_best[batch][1]:
                        
                        active_beams[batch] = False
            #print(active_beams)
            #print("look here", self.decoder.stop_index)
            #print(output)
            


            back_pointers.append(bp)
            
            #print(output)
            decoder_output_buffer[:,step].copy_(
                output.view(batch_size * beam_size))
            decoder_input_buffer[:,step + 1].copy_(
                output.view(batch_size * beam_size))
            #print(prev_state)

            import numpy as np
            has_stopped = np.argwhere(
                output[0].numpy() == self.decoder.stop_index)
            if has_stopped.shape[0] > 0:
                output_step = decoder_output_buffer.view(
                    batch_size, beam_size,-1)[:,:,:step + 1]
                #print(step)
                #print(scores)
                for s in range(has_stopped.shape[0]):
                    #print(has_stopped[s])
                    batch = has_stopped[s][0]
                    beam = has_stopped[s][1]
                    path = self.recover_path(
                        torch.cat(back_pointers, 0), output_step,
                        batch, beam)
                    #print(path)
                    score = scores.data[batch, beam]
                    if score > current_best[batch][1]:
                        current_best[batch][0] = path 
                        current_best[batch][1] = score
                    scores[batch, beam] = float("-inf")
                    current_best[batch][2].append((path, score))
                         


            #print("")
            if not np.any(active_beams):
                #print("Exiting at step", step)
                break

        batch_k_best_lists = []
        for batch in range(batch_size):
            k_best_lists = []
#            print(batch)
#            print(current_best[batch][1])
#            print(current_best[batch][0])
#                
#            print(" ".join([self.get_meta("decoder_vocab").token(idx)
#                            for idx in current_best[batch][0]]))

            current_best[batch][2].sort(key=lambda x: x[1], reverse=True)
            for tokens, score in current_best[batch][2][:10]:
                k_best_lists.append((
                    score, 
                    " ".join([self.get_meta("decoder_vocab").token(idx)
                              for idx in tokens])))
            batch_k_best_lists.append(k_best_lists)
        return batch_k_best_lists

        exit()

        print(logits)
        back_pointers = torch.cat(back_pointers, 0)
        print(scores)

        decoder_output_buffer = decoder_output_buffer.view(batch_size, beam_size, -1)[:,:,:20]
        
        seqs = self.recover_beam_path(back_pointers, 
            decoder_output_buffer)
        return seqs
    

        exit()


    def recover_path(self, back_pointers, outputs, batch, beam):
        back_pointers = back_pointers[:,batch].t()
        outputs = outputs[batch]
        max_steps = outputs.size(1)
        
        path = []
        for step in range(max_steps - 1, -1, -1):
            path.append(outputs[beam, step])
            beam = back_pointers[beam, step]  
        return path[::-1]

        



    def recover_beam_path(self, back_pointers, outputs):
        print(back_pointers)
        start_beam = 1
        batch = 0 
        batch_sequences = []
        for batch in range(outputs.size(0)):
            sequences = []
            for start_beam in range(outputs.size(1)):
                seq = []
                beam = start_beam

                for step in range(outputs.size(2)-1, -1, -1):
                    seq = [outputs[batch, beam, step]] + seq
                    new_beam = back_pointers[step,batch,beam]
                    beam = new_beam
                print(seq)
                sequences.append(seq)
            batch_sequences.append(sequences)
        return torch.LongTensor(batch_sequences)


