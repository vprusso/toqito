import numpy as np
from scipy import sparse


def iden(dim: int, is_sparse: bool) -> np.ndarray:
    if is_sparse:
        id_mat = sparse.eye(dim)
    else:
        id_mat = np.identity(dim)
    return id_mat
