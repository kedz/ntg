from attention.attention_base import AttentionBase

class NoOpAttention(AttentionBase):
    def __init__(self):
        return super(NoOpAttention, self).__init__()

    def attend(self):
        pass

    def compose(self):
        pass

    def forward(self, input, context, return_weights=True):
        if return_weights:
            return input, None
        else:
            return input
