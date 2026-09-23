import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: Integer seed value.
        deterministic: If True, sets torch cuDNN to deterministic mode.

    Returns:
        The seed integer.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed
