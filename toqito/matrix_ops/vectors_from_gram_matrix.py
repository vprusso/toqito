"""Calculates the vectors associated to a Gram matrix."""

import numpy as np
import scipy


def vectors_from_gram_matrix(gram: np.ndarray) -> list[np.ndarray]:
    r"""Obtain the corresponding ensemble of states from the Gram matrix :footcite:`WikiGram`.

    The function attempts to compute the Cholesky decomposition of the given Gram matrix. If the matrix is positive
    definite, the Cholesky decomposition is returned. If the matrix is not positive definite, the function falls back to
    eigendecomposition.

    Examples
    ========

    # Example of a positive definite matrix:

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_ops import vectors_from_gram_matrix

     gram_matrix = np.array([[2, -1], [-1, 2]])
     vectors = vectors_from_gram_matrix(gram_matrix)

     vectors

    # Example of a matrix that is not positive definite:

    .. jupyter-execute::

     gram_matrix = np.array([[0, 1], [1, 0]])
     vectors = vectors_from_gram_matrix(gram_matrix)

     vectors #Matrix is not positive semidefinite. Using eigendecomposition as alternative.

    References
    ==========
    .. footbibliography::


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
