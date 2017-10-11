from abc import abstractmethod
from models.model_base import ModelBase
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# TODO decoder should have state not prev state
# TODO implement greedy predict

class SequencePredictor(ModelBase):
    def __init__(self, input_modules, decoder):
        super(SequencePredictor, self).__init__()
        self.input_modules_ = input_modules
        self.decoder_ = decoder

    @property
    def input_modules(self):
        return self.input_modules_

    @property
    def decoder(self):
        return self.decoder_

    def forward(self, inputs, max_steps, prev_state=None, context=None):
        decoder_inputs = self.input_modules.forward_sequence(inputs, max_steps)
        logits, state = self.decoder(
            decoder_inputs, prev_state=prev_state, context=context)
        return logits, state

    def greedy_predict(self):
        pass

    def init_beam_outputs(self, prev_outputs, beam_size):
        beam_outputs = []
        for output in prev_outputs:
            if output.dim() == 1:
                batch_size = output.size(0)
                beam_output = output.contiguous().view(
                    batch_size, 1).repeat(
                    1, beam_size).view(
                    batch_size * beam_size)
                beam_outputs.append(beam_output)
            else:
                print("agh")
                exit()
        return beam_outputs

    #def init_beam_scores(self, batch_size, beam_size):
        


    def beam_step(self, log_probs, scores):
        
        batch_size = log_probs.size(0)
        beam_size = log_probs.size(1)
        
        step_scores, indices = log_probs.data.topk(beam_size, dim=2)

        step_scores.add_(
            scores.view(batch_size, beam_size, 1).repeat(
                1, 1, beam_size))
        next_scores, next_ids = step_scores.view(
                batch_size, beam_size * beam_size).topk(beam_size, dim=1)

        beam_source = next_ids.div(beam_size)
        beam_position = next_ids % beam_size

        next_output = next_ids.new(batch_size * beam_size).zero_()
        idx = 0
        for batch in range(batch_size):
            for beam in range(beam_size):
                next_output[idx] = indices[
                    batch, 
                    beam_source[batch, beam], 
                    beam_position[batch, beam]]
                idx += 1
        
        backpointers = beam_source 

        return next_output, next_scores, backpointers
        
    def backtrack(self, backpointers, outputs, batch, beam, terminal_step):
        bp = backpointers[:terminal_step + 1, batch]
        bo = outputs[:terminal_step + 1, batch]
        seq = []
        for step in range(terminal_step, -1, -1):
            token = bo[step, beam]
            beam = bp[step,beam]
            seq.append(token)
        return seq[::-1]


    # TODO change prev_outputs to inputs
    # TODO change prev_state to decoder_state
    # change this whole call signature 
    def beam_search(self, prev_outputs, prev_state=None, context=None,
                    max_steps=100, beam_size=8, beam_scores=None, 
                    batch_size=None, scores=None,
                    start_index=None, stop_index=None):
        
        beam_predictions = torch.LongTensor(
            max_steps, batch_size, beam_size).zero_()
        backpointers = torch.LongTensor(
            max_steps, batch_size, beam_size).zero_()

        # Given the backpointer table, a timestep, and a beam index we can 
        # recover a generated sequence. To return a k-bist list of completed
        # sequences we only need to rememeber when (timestep) we completed 
        # a sequence and what beam index it had.
        # for each batch keep track of best finished sequence score and 
        # all possible terminal states and their scores.
        terminal_states = [[float("-inf"), []] for batch in range(batch_size)]

        for step in range(0, max_steps):
            #print("step", step)   

            decoder_inputs = self.input_modules.forward_step(
                prev_outputs, step)     

            logits, prev_state = self.decoder(
                decoder_inputs, prev_state=prev_state, context=context)

            # TODO make sure lsm is on cpu (don't need to be on gpu for beam
            # bookkeeping)
            lsm = F.log_softmax(logits[0]).view(
                batch_size, beam_size, logits.size(2))

            next_output, next_scores, backpointer = self.beam_step(lsm, scores)
            backpointers[step].copy_(backpointer)
            beam_predictions[step].copy_(
                next_output.view(batch_size, beam_size))

            # TODO change to update_beam_decoder_state
            next_state = self.update_beam_state(
                prev_state, backpointer)
                
            prev_outputs = [Variable(next_output)] + prev_outputs[1:]
            prev_state = next_state
            scores = next_scores

            # TODO put this in a separate function
            generated_stop = beam_predictions[step].eq(stop_index).numpy()
            for batch, beam in np.argwhere(generated_stop):
                score = scores[batch, beam]
                best_score = terminal_states[batch][0]
                if score > best_score:
                    terminal_states[batch][0] = score
                terminal_states[batch][1].append((score, step, beam))
                # prevent further exploration of a terminal sequence
                scores[batch, beam] = float("-inf")
                
            still_active = [scores[batch][0] > terminal_states[batch][0] 
                            for batch in range(batch_size)]
            #print(still_active)
            if not any(still_active):
                break

        results = []
        for batch in range(batch_size):
            terminal_states[batch][1].sort(key=lambda x: x[0], reverse=True)
            result = []    
            for score, step, beam in terminal_states[batch][1][:10]:
                #print(batch, score, step, beam)
                sequence = self.backtrack(
                    backpointers, beam_predictions, batch, beam, step) 
                result.append((score, sequence))
            results.append(result)
        return results
