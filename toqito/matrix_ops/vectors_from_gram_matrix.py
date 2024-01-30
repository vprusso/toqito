"""Vectors associated to Gram matrix."""
import numpy as np
import scipy


def vectors_from_gram_matrix(gram: np.ndarray) -> list[np.ndarray]:
    r"""Obtain the corresponding ensemble of states from the Gram matrix :cite:`WikiGram`.

    The function attempts to compute the Cholesky decomposition of the given Gram matrix. If the matrix is positive
    definite, the Cholesky decomposition is returned. If the matrix is not positive definite, the function falls back to
    eigendecomposition.

    Examples
    ========

    # Example of a positive definite matrix:

    >>> gram_matrix = np.array([[2, -1], [-1, 2]])
    >>> vectors = vectors_from_gram_matrix(gram_matrix)
    >>> vectors
    [array([1.41421356, 0.        ]), array([-0.70710678,  1.22474487])]

    # Example of a matrix that is not positive definite:

    >>> gram_matrix = np.array([[0, 1], [1, 0]])
    >>> vectors = vectors_from_gram_matrix(gram_matrix)

    Matrix is not positive semidefinite. Using eigendecomposition as alternative.
    >>> vectors
    [array([0.70710678+0.j, 0.        +0.j]), array([0.        +0.j, 0.70710678+0.j])]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :raises LinAlgError: If the Gram matrix is not square.
    :param gram: A square, symmetric matrix representing the Gram matrix.
    :return: A list of vectors (np.ndarray) corresponding to the ensemble of states.

    """
    dim = gram.shape[0]
    if gram.shape[0] != gram.shape[1]:
        raise np.linalg.LinAlgError("The Gram matrix must be square.")

    # If matrix is PD, can do Cholesky decomposition:
    try:
        decomp = np.linalg.cholesky(gram)
        return [decomp[i][:] for i in range(dim)]
    # Otherwise, need to do eigendecomposition:
    except np.linalg.LinAlgError:
        print("Matrix is not positive semidefinite. Using eigendecomposition as alternative.")
        d, v = np.linalg.eig(gram)
        return [scipy.linalg.sqrtm(np.diag(d)) @ v[i].conj().T for i in range(dim)]
