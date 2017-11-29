import torch
import torch.nn.functional as F
from attention.attention_base import AttentionBase


class DotAttention(AttentionBase):
    def __init__(self, merge_type="concat"):
        super(DotAttention, self).__init__(merge_type=merge_type)

    def attend(self, input, context):
        batch_size = input.size(1)
        hidden_size = input.size(2)
        input_steps = input.size(0)
        context_steps = context.size(0)        
        
        score = torch.bmm(input.transpose(1, 0), context.permute(1, 2, 0))
        score.data.masked_fill_(score.data.eq(0), float("-inf"))
        score_flat = score.view(batch_size * input_steps, context_steps)
        
        weight_flat = F.softmax(score_flat)
        weight = weight_flat.view(batch_size, input_steps, context_steps)
        return weight
