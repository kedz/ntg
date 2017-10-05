import torch
import torch.nn as nn


class AttentionBase(nn.Module):
    def __init__(self, merge_type="concat"):
        super(AttentionBase, self).__init__()
        self.merge_type_ = merge_type

        if self.merge_type == "concat":
            self.merge = self.concat_merge_ 
        else:
            raise Exception(
                "merge_type {} not implemented.".format(self.merge_type))

    @property
    def merge_type(self):
        return self.merge_type_

    def attend(self, input, context):
        raise Exception("Not implemented.")

    def compose(self, context, weights):
        composed_context_T = torch.bmm(weights, context.permute(1, 0, 2))
        composed_context = composed_context_T.permute(1, 0, 2)
        return composed_context

    def concat_merge_(self, input, composed_context):
        output = torch.cat([input, composed_context], 2)
        return output

    def forward(self, input, context):
        weights = self.attend(input, context)
        composed_context = self.compose(context, weights)
        output = self.merge(input, composed_context)
        return output, weights
