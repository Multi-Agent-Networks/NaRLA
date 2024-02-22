import random

import numpy as np
import torch


def reproducibility(seed: int):
    """
    Set backend reproducibility with random seeds

    :param seed: Seed to set in ``random, numpy, torch``
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
