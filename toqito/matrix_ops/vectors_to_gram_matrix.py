"""Gram matrix from list of vectors."""
import numpy as np


def vectors_to_gram_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    r"""Construct the Gram matrix from a list of vectors :cite:`WikiGram`.

    The Gram matrix is a matrix of inner products, where the entry G[i, j] is the inner product of vectors[i] and
    vectors[j]. This function computes the Gram matrix for a given list of vectors.

    Examples
    ========
    >>> # Example with real vectors:
    >>> vectors = [np.array([1, 2]), np.array([3, 4])]
    >>> gram_matrix = vectors_to_gram_matrix(vectors)
    >>> gram_matrix
    array([[ 5., 11.],
           [11., 25.]])

    >>> # Example with complex vectors:
    >>> vectors = [np.array([1+1j, 2+2j]), np.array([3+3j, 4+4j])]
    >>> gram_matrix = vectors_to_gram_matrix(vectors)
    >>> gram_matrix
    array([[ 10.+0.j,  20.+0.j],
           [ 20.+0.j,  40.+0.j]])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :raises ValueError: If the vectors are not all of the same length.
    :param vectors: A list of vectors (1D numpy arrays). All vectors must be of the same length.
    :return: A list of vectors corresponding to the ensemble of states.

    """
    # Check that all vectors are of the same length
    if not all(v.shape == vectors[0].shape for v in vectors):
        raise ValueError("All vectors must be of the same length.")

    # Stack vectors into a matrix
    stacked_vectors = np.column_stack(vectors)

    # Compute Gram matrix using vectorized operations
    return np.dot(stacked_vectors.conj().T, stacked_vectors)
