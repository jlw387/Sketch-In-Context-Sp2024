"""
Unused (I think) file for vector math (the only thing implemented in normalization, so I doubt I used it anywhere)."""

import numpy as np

def normalize(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return 0
    try:
        ret = x / norm
        return ret
    except:
        print("X:",x)
        print("Norm:",norm)
