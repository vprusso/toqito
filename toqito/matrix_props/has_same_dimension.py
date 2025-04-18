"""Checks if the dimensions of list of vectors or matrices are equal."""

import numpy as np


def has_same_dimension(items: list[np.ndarray]) -> bool:
    """Check if all vectors or matrices in a list have the same dimension.

    For a vector (1D array), the dimension is its length. For a matrix, the dimension can be considered as the total
    number of elements (rows x columns) for non-square matrices, or simply the number of rows (or columns) for square
    matrices. The function iterates through the provided list and ensures that every item has the same dimension.

    Examples
    ==========
    Check a list of vectors with the same dimension:

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import has_same_dimension

     vectors = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

     has_same_dimension(vectors)


    Check a list of matrices with the same dimension:

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import has_same_dimension

     matrices = [np.array([[1, 0], [0, 1]]), np.array([[2, 3], [4, 5]]), np.array([[6, 7], [8, 9]])]

     has_same_dimension(matrices)

    Check a list containing items of different dimensions:

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import has_same_dimension

     mixed = [np.array([1, 2, 3]), np.array([[1, 0], [0, 1]])]

     has_same_dimension(mixed)


    :param items: A list containing vectors or matrices. Vectors are represented as 1D numpy arrays, and matrices are
                  represented as 2D numpy arrays.
    :return: Returns :code:`True` if all items in the list have the same dimension, :code:`False` otherwise.
    :raises ValueError: If the input list is empty.

    """
    if len(items) == 0:
        raise ValueError("The list is empty.")

    first_item = items[0]
    # Checking for numpy array to handle matrix case
    if isinstance(first_item[0], np.ndarray):
        expected_dim = len(first_item) * len(first_item[0])
    else:
        expected_dim = len(first_item)

    for item in items[1:]:
        if isinstance(item[0], np.ndarray):
            dim = len(item) * len(item[0])
        else:
            dim = len(item)

        if dim != expected_dim:
            return False
    return True
