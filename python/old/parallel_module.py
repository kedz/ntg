import torch.nn as nn

class ParallelModule(nn.Module):
    def __init__(self, module_list):
        super(ParallelModule, self).__init__()
        self.module_list_ = nn.ModuleList(module_list)
    
    @property
    def module_list(self):
        return self.module_list_

    def forward(self, inputs):
        return tuple([module(input) 
                      for input, module in zip(inputs, self.module_list)])
