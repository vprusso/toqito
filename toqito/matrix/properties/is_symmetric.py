import numpy as np


def is_symmetric(mat: np.ndarray,
                 rtol: float = 1e-05,
                 atol: float = 1e-08) -> bool:
    return np.allclose(mat, mat.T, rtol=rtol, atol=atol)
