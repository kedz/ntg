import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from ntp.functional import log_sum_exp


class LinearChainDecoder(nn.Module):

    def __init__(self, num_states):
        super(LinearChainDecoder, self).__init__()
        self.num_states_ = num_states
        self.start_factors_ = nn.Parameter(
            torch.randn(num_states))
        self.stop_factors_ = nn.Parameter(
            torch.randn(num_states))
        self.transition_factors_ = nn.Parameter(
            torch.randn(num_states, num_states))

    @property
    def num_states(self):
        return self.num_states_

    @property
    def start_factors(self):
        return self.start_factors_

    @property
    def stop_factors(self):
        return self.stop_factors_

    @property
    def transition_factors(self):
        return self.transition_factors_

    def _get_sizes_safe(self, sizes_list_tuple_var_or_tensor):
        if isinstance(sizes_list_tuple_var_or_tensor, (list, tuple)):
            sizes = sizes_list_tuple_var_or_tensor
        elif isinstance(sizes_list_tuple_var_or_tensor, Variable):
            sizes = sizes_list_tuple_var_or_tensor.data.tolist()
        elif torch.is_tensor(sizes_list_tuple_var_or_tensor):
            sizes = sizes_list_tuple_var_or_tensor.tolist()
        else:
            raise Exception(
                "Argument must be list, tuple, Variable, or Tensor")
        return sizes

    def pack_sequence(self, sequences, sequence_sizes=None): 
        if isinstance(sequences, torch.nn.utils.rnn.PackedSequence):
            # Emission factors have already been placed in a packed sequence.
            # No need to perform checks or packing.
        
            sequences_flat = sequences.data
            batch_sizes = sequences.batch_sizes

        elif sequence_sizes is None:
            # Assume all batch items have the same number of time steps.

            batch_size = sequences.size(1)
            max_steps = sequences.size(0)
            batch_sizes = [batch_size] * max_steps
            sequences_flat = sequences.contiguous().view(
                batch_size * max_steps, -1)

        else:
            # Expect that sequence_sizes is not None and prepare to check
            # size order is descending and pack emission factors.

            sequence_sizes = self._get_sizes_safe(sequence_sizes)
            pack = pack_padded_sequence(sequences, sequence_sizes)
            sequences_flat = pack.data
            batch_sizes = pack.batch_sizes

        return sequences_flat, batch_sizes

    def prepare_step_iter(self, emission_factors, sequence_sizes=None,
                          state_sequence=None):

        emissions_flat, batch_sizes = self.pack_sequence(
            emission_factors, sequence_sizes=sequence_sizes)

        if state_sequence is not None:
            state_sequences_flat, _ = self.pack_sequence(
                state_sequence, sequence_sizes=sequence_sizes)
        else:
            state_sequences_flat = None

        max_steps = len(batch_sizes)
        max_batch_size = batch_sizes[0]

        def step_iter():
            
            batch_start = prev_batch_size = batch_sizes[0]
            max_steps = len(batch_sizes)

            start_ftr = self.start_factors.view(1, -1).repeat(
                batch_sizes[0], 1)

            trans_ftr = self.transition_factors.view(
                1, self.num_states, self.num_states).repeat(
                    batch_sizes[0], 1, 1)

            if state_sequences_flat is not None:

                yield (0, batch_start, start_ftr, emissions_flat[:batch_start],
                    state_sequences_flat[:batch_start])

            else:
                yield 0, batch_start, start_ftr, emissions_flat[:batch_start]

            for step, batch_size in enumerate(batch_sizes[1:], 1):

                batch_stop = batch_start + batch_size
                
                emissions_step = emissions_flat[batch_start:batch_stop]

                if batch_size != prev_batch_size:
                    trans_ftr = self.transition_factors.view(
                        1, self.num_states, self.num_states).repeat(
                            batch_size, 1, 1)
                
                if state_sequences_flat is not None:
                    state_seq = state_sequences_flat[batch_start:batch_stop]
                    yield (step, batch_size, trans_ftr, emissions_step, 
                        state_seq)
                else:
                    yield step, batch_size, trans_ftr, emissions_step

                batch_start = batch_stop                
                prev_batch_size = batch_size

        return max_steps, max_batch_size, step_iter()


    def _get_paths_from_backpointers(self, bp_scores, bp_states):
        first_batch_size = bp_scores[0].size(0)
        max_steps = len(bp_scores)
        viterbi_scores = []
        viterbi_paths = [None] * max_steps
        prev_batch_size = 0
        prev_index = Variable(bp_scores[max_steps - 1].data.new())

        for step in range(max_steps - 1, -1, -1):
            cur_batch_size = bp_scores[step].size(0)

            if cur_batch_size > prev_batch_size:
                next_viterbi_scores, next_end_states = torch.max(
                    bp_scores[step][prev_batch_size:], 1)

                viterbi_scores.append(next_viterbi_scores)

                if prev_index.numel() > 0:
                    prev_index = torch.cat([prev_index, next_end_states])
                else:
                    prev_index = next_end_states

            if first_batch_size > cur_batch_size:
                pad_size = first_batch_size - cur_batch_size
                pad = Variable(prev_index.data.new([-1] * pad_size))
                viterbi_paths[step] = torch.cat([prev_index, pad]).view(-1, 1)
            else:
                viterbi_paths[step] = prev_index.view(-1, 1)

            prev_index = bp_states[step].gather(
                1, prev_index.unsqueeze(1)).view(-1)

            prev_batch_size = cur_batch_size

        return torch.cat(viterbi_paths, 1), torch.cat(viterbi_scores)

    def viterbi_decode(self, emission_factors, sequence_sizes=None,
                       return_score=False):

        max_steps, max_batch_size, step_iter = self.prepare_step_iter(
            emission_factors, sequence_sizes=sequence_sizes)

        bp_scores = [None] * max_steps
        bp_states = [None] * max_steps

        _, first_batch_size, start_ftr, emission_ftr = next(step_iter)

        prev_batch_size = first_batch_size
        bp_scores[0] = start_ftr + emission_ftr
        
        if emission_factors.is_cuda:
            bp_states[0] = Variable(
                torch.cuda.LongTensor(
                    max_batch_size, self.num_states).fill_(-1))
        else:
            bp_states[0] = Variable(
                torch.LongTensor(
                    max_batch_size, self.num_states).fill_(-1))

        batch_size = first_batch_size

        for step, batch_size, trans_ftr, emission_ftr in step_iter:

            if batch_size != prev_batch_size:
                zeros_part = Variable(
                    self.stop_factors.data.new(
                        batch_size, self.num_states).fill_(0))
                stop_factors_part = self.stop_factors.view(1, -1).repeat(
                    prev_batch_size - batch_size, 1)

                stop_factors = torch.cat([zeros_part, stop_factors_part], 0)
                bp_scores[step - 1] += stop_factors
            
            prev_score_mat = bp_scores[step-1].view(
                prev_batch_size, 1, self.num_states).repeat(
                    1, self.num_states, 1)
            trans_mat = prev_score_mat[:batch_size] + trans_ftr
            
            max, argmax = torch.max(trans_mat, 2)
            
            bp_scores[step] = max + emission_ftr
            bp_states[step] = argmax
            prev_batch_size = batch_size

        bp_scores[-1] += self.stop_factors.view(
            1, -1).repeat(batch_size, 1)

        viterbi_paths, viterbi_scores = self._get_paths_from_backpointers(
            bp_scores, bp_states)            

        if return_score:
            return viterbi_paths, viterbi_scores
        else:
            return viterbi_paths

    def forward_algorithm(self, emission_factors, sequence_sizes=None):

        max_steps, max_batch_size, step_iter = self.prepare_step_iter(
            emission_factors, sequence_sizes=sequence_sizes)

        _, first_batch_size, start_ftr, emission_ftr = next(step_iter)

        prev_batch_size = first_batch_size
        alphas = start_ftr + emission_ftr
        penultimate_alphas = []

        for step, batch_size, trans_ftr, emission_ftr in step_iter:
            
            if batch_size != prev_batch_size:
                # add stop transition to sequences that were finished last 
                # step
                penultimate_alphas.append(alphas[-prev_batch_size+batch_size:])
                
            alphas_3d = alphas.view(
                prev_batch_size, 1, self.num_states).repeat(
                        1, self.num_states, 1)[:batch_size]

            trans_mat_3d = alphas_3d + trans_ftr
            alphas = log_sum_exp(trans_mat_3d, 2).add_(emission_ftr)
            prev_batch_size = batch_size

        penultimate_alphas.append(alphas)

        penultimate_alphas = torch.cat(penultimate_alphas[::-1], 0)
        stop_ftr = self.stop_factors.view(1, -1).repeat(first_batch_size, 1)
        final_alphas = penultimate_alphas + stop_ftr

        log_normalizer = log_sum_exp(final_alphas, 1)
        return log_normalizer

    def score_state_sequence(self, emission_factors, state_sequences,
                             sequence_sizes=None):

        max_steps, max_batch_size, step_iter = self.prepare_step_iter(
            emission_factors, sequence_sizes=sequence_sizes, 
            state_sequence=state_sequences)

        _, first_batch_size, start_ftr, emission_ftr, states = next(step_iter)

        final_scores = []
        scores = start_ftr.gather(1, states.view(-1,1)) + \
            emission_ftr.gather(1, states.view(-1,1))

        prev_batch_size = first_batch_size
        prev_states = states

        for step, batch_size, trans_ftr, emission_ftr, states in step_iter:

            if batch_size != prev_batch_size:
                stop_trans = self.stop_factors.index_select(
                    0, prev_states[batch_size:]).view(-1, 1)
                final_scores.append(scores[batch_size:] + stop_trans)
                scores = scores[:batch_size]

            trans_ftr = [trans_ftr[b,states.data[b],prev_states.data[b]]
                         for b in range(batch_size)]
            trans_ftr = torch.cat(trans_ftr)
            scores = scores + trans_ftr.view(-1, 1) + \
                emission_ftr.gather(1, states.view(-1, 1))

            prev_batch_size = batch_size
            prev_states = states

        stop_trans = self.stop_factors.index_select(
            0, prev_states).view(-1, 1)
        final_scores.append(scores + stop_trans)
        return torch.cat([score.view(-1) for score in final_scores[::-1]], 0)
