import random

import torch
import numpy as np
from accelerate.utils import set_seed

def freeze_random(seed: int):
    """
    Freeze random state with a given seed
    for reproducability.
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True