import torch
import torch.nn as nn

# TODO create separate sequence layer norm.
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.features_ = features
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        view_dims = [1] * (x.dim() - 1) + [self.features]
        repeat_dims = [z for z in x.size()[:-1]] + [1]
        gamma = self.gamma.view(*view_dims).repeat(*repeat_dims)
        beta = self.beta.view(*view_dims).repeat(*repeat_dims)

        return gamma * (x - mean) / (std + self.eps) + beta

    @property
    def features(self):
        return self.features_
