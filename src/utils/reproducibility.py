"""Reproducibility utilities"""

import random
import numpy as np
import os


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Torch seeds (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
