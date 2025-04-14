"""Computes the mutual coherence of the columns of a matrix or a list of 1D numpy arrays."""

import numpy as np


def mutual_coherence(matrix: np.ndarray) -> float:
    r"""Calculate the mutual coherence of a set of states.

    The mutual coherence of a matrix is defined as the maximum absolute value
    of the inner product between any two distinct columns, divided
    by the product of their norms. The mutual coherence is a measure of how
    distinct the given columns are.

    Note: As mutual coherence is also useful in the context of quantum states,
    a list of 1D numpy arrays is also accepted as input.

    Examples
    ==========
    >>> import numpy as np
    >>> from toqito.matrix_props.mutual_coherence import mutual_coherence
    >>> matrix_A = np.array([[1, 0], [0, 1]])
    >>> mutual_coherence(matrix_A)
    0.0

    >>> # An example with a larger matrix
    >>> matrix_B = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    >>> mutual_coherence(matrix_B)
    1/2

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param states: A 2D numpy array or a list of 1D
            numpy arrays representing a set of states (list[np.ndarray] or np.ndarray).
    :raises isinstance: Check if input is valid.
    :return: The mutual coherence.

    """
    # Check if the input is a valid numpy array or a list of 1D numpy arrays
    if not isinstance(matrix, (np.ndarray, list)):
        raise TypeError("Input must be a numpy array or a list of 1D numpy arrays.")

    # If input is a list of 1D arrays, convert it to a 2D numpy array
    if isinstance(matrix, list):
        matrix = np.column_stack(matrix)

    # Normalize the states
    matrix = matrix / np.linalg.norm(matrix, axis=0)

    # Calculate the inner product between all pairs of columns
    inner_products = np.abs(np.conj(matrix.T) @ matrix)

    # Set diagonal elements to zero (self-inner products)
    np.fill_diagonal(inner_products, 0)

    # Calculate mutual coherence
    mutual_coherence = inner_products.max()

    return mutual_coherence
