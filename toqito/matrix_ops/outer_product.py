"""Outer product operation"""
import numpy as np

def outer_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    
    # Check for dimensional validity
    if not (v1.shape[0] == v2.shape[0] and v1.shape[0] > 1 and v1.shape[1] == v2.shape[1] == 1):
        raise ValueError("Dimension mismatch")

    res = np.ndarray((v1.shape[0], v1.shape[0]))
    for i in range(v1.shape[0]):
        for j in range(v1.shape[0]):
            res[i,j] = v1[i,0] * v2[j,0]
    return res