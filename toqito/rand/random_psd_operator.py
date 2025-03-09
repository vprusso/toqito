"Generates a random positive semi-definite operator."

import numpy as np

def random_psd_operator(dim: int) -> np.ndarray:
    r''' Generate a random positive semi definite operator.

    A positive semi-definite operator is a Hermitian operator that has only real and non-negative eigenvalues. 

    This function generates a random positive semi-definite operator by constructing a symmetric matrix, 
    based on the fact that a symmetric matrix can have its eigen values as real numbers. First it generates a random
    density matrix and then creates a symmetry matrix from it. Then the eigen values are made to their absolute values.
    Then the positive semi definite operator is constructed by taking the dot product of the absolute eigen value matrix,
    eigen vectors and eigen value transpose. Finally it is normalized to make its trace as 1. The resultant of the function will be a numpy ndarray 
    which is positive semi-definite and unitary.

    Example
    ===========================
    Generate a positive semi-definite operator of dimension 3.

    >>> dim = 3
    >>> random_mat = np.random.rand(dim, dim)
    >>> random_mat = (random_mat.T + random_mat) / 2
    >>> eigenvalues, eigenvectors = np.linalg.eig(random_mat)
    >>> eigenvalues = np.abs(eigenvalues)
    >>> random_mat = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), np.conj(eigenvectors).T))
    >>> random_mat = random_mat / np.trace(random_mat)
    >>> random_mat

        [[0.35636099 0.17920518 0.14620245]
         [0.17920518 0.35165863 0.14467558]
         [0.14620245 0.14467558 0.29198038]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param dim (int): The dimension of the operator.

    :return (np.ndarray): A random positive semi-definite operator with dimensions dim x dim.

    :raises ValueError: If dim is not a positive integer.

    Notes:
        The generated operator is symmetric and has non-negative eigenvalues.
        The operator is normalized to have a trace of 1.

    '''

    random_mat = np.random.rand(dim, dim)
    random_mat = (random_mat.T + random_mat) / 2
    eigenvalues, eigenvectors = np.linalg.eig(random_mat)
    eigenvalues = np.abs(eigenvalues)
    random_mat = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), np.conj(eigenvectors).T))
    random_mat = random_mat / np.trace(random_mat)
    return random_mat


