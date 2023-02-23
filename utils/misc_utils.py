"""
Miscellaneous utility functions
"""
import numpy as np

import random

import torch


def set_seeds(seed):
    """
    Sets the seed for random, NumPy and PyTorch
    """
    # Back to random seeds
    random.seed(seed)
    np.random.seed(seed)

    # For PyTorch & CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False