import torch
import torch.nn as nn
from torch.autograd import Variable


class NeuralCRFTagger(nn.Module):
    def __init__(self, input_module, feature_module, emission_module,
                 pad_value=0):
        super(NeuralCRFTagger, self).__init__()

        self.input_module_ = input_module
        self.feature_module_ = feature_module
        self.emission_module_ = emission_module 
        self.pad_value_ = pad_value
        self.num_tags_ = emission_module.output_size
        self.transition_params_ = nn.Parameter(
            torch.randn(self.num_tags, self.num_tags))
        self.start_transition_params_ = nn.Parameter(
            torch.randn(1, self.num_tags))
        self.stop_transition_params_ = nn.Parameter(
            torch.randn(1, self.num_tags))

    @property
    def start_index(self):
        return self.start_index_

    @property
    def stop_index(self):
        return self.stop_index_

    @property
    def num_tags(self):
        return self.num_tags_

    @property
    def pad_value(self):
        return self.pad_value_

    @property
    def input_module(self):
        return self.input_module_

    @property
    def feature_module(self):
        return self.feature_module_

    @property
    def emission_module(self):
        return self.emission_module_

    @property
    def transition_params(self):
        return self.transition_params_

    @property
    def start_transition_params(self):
        return self.start_transition_params_

    @property
    def stop_transition_params(self):
        return self.stop_transition_params_

    def compute_emissions(self, inputs, mask=None):
        sequence, length = inputs
        batch_size = sequence.size(0)
        sequence_size = sequence.size(1)

        if mask is None:
            mask = sequence.eq(self.pad_value)
        
        emissions_mask = mask.view(batch_size, sequence_size, 1).repeat(
            1, 1, self.num_tags).transpose(1, 0)

        encoder_input = self.input_module(sequence)
        context_sequence = self.feature_module.encoder_context(
            encoder_input, length=length)
        context_flat = context_sequence.view(sequence_size * batch_size, -1)

        emissions_flat = self.emission_module(context_flat)
        emissions = emissions_flat.view(
            sequence_size, batch_size, -1).masked_fill(
            emissions_mask, 0)
        return emissions

    def log_normalizer(self, emissions, mask):

        batch_size = emissions.size(1)
        sequence_size = emissions.size(0)
        
        mask_4d = mask.view(sequence_size, batch_size, 1, 1).repeat(
            1, 1, self.num_tags, self.num_tags)
        
        alphas = Variable(
            torch.FloatTensor(batch_size, self.num_tags).fill_(0)) 

        alphas = alphas + self.start_transition_params.repeat(batch_size, 1)
        #alphas[:,2] = 1
        for step in range(sequence_size):
        #    print(alphas)
         #   print(alphas.view(batch_size, 1, self.num_tags).repeat(
         #       1, self.num_tags, 1))
         #   print(emissions[step].view(batch_size, self.num_tags, 1).repeat(
          #      1, 1, self.num_tags))
          #  print(self.transition_params.view(
         #       1, self.num_tags, self.num_tags).repeat(batch_size, 1, 1))
            alphas_3d = alphas.view(
                batch_size, 1, self.num_tags).repeat(1, self.num_tags, 1)
            emissions_3d_step = emissions[step].view(
                batch_size, self.num_tags, 1).repeat(1, 1, self.num_tags)
            trans_3d = self.transition_params.view(
                1, self.num_tags, self.num_tags).repeat(batch_size, 1, 1)
            trans_3d = trans_3d.masked_fill(mask_4d[step], 0)
            scores_3d = alphas_3d + trans_3d + emissions_3d_step
            alphas = scores_3d.sum(2) # should do log sum exp
        
        terminal_alphas = alphas + self.stop_transition_params.repeat(
            batch_size, 1)
        log_normalizer = terminal_alphas.sum(1) # should do log sum exp
        return log_normalizer

    def sequence_log_score(self, emissions, targets):

        mask = targets.eq(-1)
        targets = targets.masked_fill(mask, 0)
        batch_size = emissions.size(1)
        sequence_size = emissions.size(0)

        trans_params = self.transition_params.view(
            self.num_tags, self.num_tags)
       
        start_trans = self.start_transition_params.view(-1)
        log_scores = start_trans.index_select(0, targets[:,0]) \
            + emissions[0].gather(1, targets[:,0:1]).view(batch_size)

        for step in range(1, sequence_size):
            
            #print(trans_params)
            #print(targets[:,step])
            #print(trans_params.index_select(0, targets[:,step]))

            # batch x num tags, ith entry of dim 1 is the probability of 
            # transitioning to the current tag from tag i
            batch_trans_step = trans_params.index_select(0, targets[:,step])

            batch_trans_score = batch_trans_step.gather(
                1, targets[:,step-1:step]).view(batch_size)
            print(step)
            print(log_scores)
            print(batch_trans_score)
            batch_emission_score = emissions[step].gather(
                1, targets[:,step:step+1]).view(batch_size)
            print(batch_emission_score)
            print(mask[:,step])
            log_scores = log_scores + batch_trans_score + batch_emission_score
            

        exit()

    def forward(self, inputs, targets, mask=None):

        emissions = self.compute_emissions(inputs)
        mask = inputs.sequence.transpose(1, 0).eq(self.pad_value)
        log_Z = self.log_normalizer(emissions, mask)
        self.sequence_log_score(emissions, targets) 
