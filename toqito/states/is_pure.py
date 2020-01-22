from numpy import linalg as LA
import numpy as np


def is_pure(state: np.ndarray) -> bool:
    eigs, _ = LA.eig(state)
    return np.isclose(np.max(np.diag(eigs)), 1)
