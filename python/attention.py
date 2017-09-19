import torch
import torch.nn as nn
import torch.nn.functional as F


class DotAttention(nn.Module):

    def __init__(self, context_dropout=0.0):
        super(DotAttention, self).__init__()

        self.context_dropout_ = context_dropout

    def forward(self, context, target):

        context, length = nn.utils.rnn.pad_packed_sequence(
            context, batch_first=True)

        context = F.dropout(
            context, p=self.context_dropout_, 
            training=self.training, inplace=False)

        score = torch.bmm(
            target.transpose(1, 0), context.transpose(2,1))
        score.data.masked_fill_(score.data.eq(0), float("-inf"))

        score_flat = score.view(score.size(0) * score.size(1), score.size(2))
        attn_flat = F.softmax(score_flat)
        attn = attn_flat.view(score.size(0), score.size(1), score.size(2))

        result = torch.bmm(attn, context).transpose(1, 0)

        return attn.transpose(1, 0), result
