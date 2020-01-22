import numpy as np


def vec(mat: np.ndarray):
    return mat.reshape((-1, 1), order="F")
