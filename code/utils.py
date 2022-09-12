import torch
import numpy as np

def to_torch(a):
    if isinstance(a, torch.Tensor):
        return a

    if isinstance(a, np.ndarray):
        return torch.from_numpy(a).float()

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Helper class for easy object access
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
