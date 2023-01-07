"""Inner product operation"""
import numpy as np

def inner_product(v1: np.ndarray, v2: np.ndarray) -> float:

    # Check for dimensional validity
    if not (v1.shape[0] == v2.shape[0] and v1.shape[0] > 1 and v1.shape[1] == v2.shape[1] == 1):
        raise ValueError("Dimension mismatch")
    
    res = 0
    for i in range(v1.shape[0]):
        res += v1[i,0] * v2[i,0]
    
    return res