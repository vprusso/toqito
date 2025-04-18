"""Calculates the (common) dimension of a set of vectors or matrices."""

import numpy as np


def calculate_vector_matrix_dimension(item: np.ndarray) -> int:
    r"""Calculate the dimension of a vector or a square matrix, including 2D representations of vectors.

    This function determines the dimension of the provided item, treating 1D arrays as vectors,
    2D arrays with one dimension being 1 as vector representations, and square 2D arrays as density matrices.
    The dimension is the length for vectors and the square of the side length for density matrices.


    :param item: The item whose dimension is being calculated. Can be a 1D array (vector), a 2D array representing
                 a vector with one dimension being 1, or a square 2D array (density matrix).
    :return: int
        The dimension of the item. For vectors (1D or 2D representations), it's the length. For square
        matrices, it's the square of the size of one side.
    :raises ValueError:
        If the input is not a numpy array, not a 1D array (vector), a 2D array representing a vector, or a square 2D
        array (density matrix).
    :return: The dimension of the vector or matrix.

    Example:
    ==========

    Consider the following three-dimensional vector:

    .. math::
        v = \left[ 1, 0, 0 \right]^{\text{T}}.

    For this case, the dimension of the vector is equal to its length

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_ops import calculate_vector_matrix_dimension

     v = np.array([1, 0, 0])

     calculate_vector_matrix_dimension(v)

    For the density matrix of some two-dimensional quantum system

    .. math::
        \rho = \frac{1}{2}
                \begin{pmatrix}
                    1 & 0 \\
                    0 & 1
                \end{pmatrix}

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_ops import calculate_vector_matrix_dimension

     rho = np.array([[1/2, 0],[0, 1/2]])

     calculate_vector_matrix_dimension(rho)

    """
    # Check if the input is a numpy array
    if not isinstance(item, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    if item.ndim == 1:
        return item.size
    if item.ndim == 2:
        if item.shape[0] == 1 or item.shape[1] == 1:
            return max(item.shape)
        if item.shape[0] == item.shape[1]:
            return item.shape[0]
        raise ValueError("Input must be either a vector or a square matrix.")
    raise ValueError("Input must be either a vector or a square matrix.")
