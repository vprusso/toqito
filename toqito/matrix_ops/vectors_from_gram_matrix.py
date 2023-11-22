"""Vectors associated to Gram matrix."""
import numpy as np


def vectors_from_gram_matrix(gram: np.ndarray) -> list[np.ndarray]:
    """Obtain the corresponding ensemble of states from the Gram matrix.


    :param gram: Input Gram matrix.
    :return: list of ensemble states
    """
    dim = gram.shape[0]
    # If matrix is PD, can do Cholesky decomposition:
    try:
        decomp = np.linalg.cholesky(gram)
        return [decomp[i][:] for i in range(dim)]
    # Otherwise, need to do eigendecomposition:
    except np.linalg.LinAlgError:
        print("Matrix is not positive semidefinite. Using eigendecomposition as alternative.")
        d, v = np.linalg.eig(gram)
        return [np.sqrt(np.diag(d)) @ v[i].conj().T for i in range(dim)]
