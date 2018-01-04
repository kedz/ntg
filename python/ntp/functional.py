import torch


def log_sum_exp(tensor, dim=None, keepdim=False):
    if dim is None:
        max = torch.max(tensor)
        return torch.log(torch.sum(torch.exp(tensor - max))) + max
    else:
        max, argmax = torch.max(tensor, dim, keepdim=True)
        exp_values = torch.exp(tensor - max)
        if not keepdim:            
            max = max.squeeze(dim)
        sum = torch.sum(exp_values, dim, keepdim=keepdim)
        return torch.log(sum).add_(max)
