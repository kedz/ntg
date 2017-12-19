import torch
import random
import numpy as np


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # TODO figure out if this is superfluous
    random.seed(seed)
    np.random.seed(seed)
