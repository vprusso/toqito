"""Convert row or column vector to density matrix."""

import numpy as np


def to_density_matrix(input_array: np.ndarray) -> np.ndarray:
    """Convert a given vector to a density matrix or return the density matrix if already given.

    If the input is a vector, this function computes the outer product to form a density matrix.
    If the input is already a density matrix (square matrix), it returns the matrix as is.

    Examples
    ==========

    As an example, consider one of the Bell states.

    >>> from toqito.states import bell
    >>> from toqito.matrix_ops import to_density_matrix
    >>>
    >>> to_density_matrix(bell(0))
    array([[0.5, 0. , 0. , 0.5],
           [0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. ],
           [0.5, 0. , 0. , 0.5]])


    :raises ValueError: If the input is not a vector or a square matrix.
    :param input_array: Input array which could be a vector or a density matrix.
    :return: The computed or provided density matrix.

    """
    # Check if the input is a vector (1D array) or a 2D array
    if input_array.ndim == 1:
        # Input is a vector, compute the density matrix
        density_matrix = np.outer(input_array, np.conjugate(input_array))
    elif input_array.ndim == 2:
        # Flatten the array if it's a column vector (n, 1) or a row vector (1, n)
        if input_array.shape[0] == 1 or input_array.shape[1] == 1:
            vector = input_array.flatten()
            density_matrix = np.outer(vector, np.conjugate(vector))
        elif input_array.shape[0] == input_array.shape[1]:
            # Input is a square matrix, assumed to be a density matrix, return as is
            density_matrix = input_array
        else:
            raise ValueError("Input must be either a vector or a square density matrix.")
    else:
        raise ValueError("Input must be either a vector or a square density matrix.")

    return density_matrix
