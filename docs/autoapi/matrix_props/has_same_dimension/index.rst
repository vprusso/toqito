matrix_props.has_same_dimension
===============================

.. py:module:: matrix_props.has_same_dimension

.. autoapi-nested-parse::

   Checks if the dimensions of list of vectors or matrices are equal.



Functions
---------

.. autoapisummary::

   matrix_props.has_same_dimension.has_same_dimension


Module Contents
---------------

.. py:function:: has_same_dimension(items)

   Check if all vectors or matrices in a list have the same dimension.

   For a vector (1D array), the dimension is its length. For a matrix, the dimension can be considered as the total
   number of elements (rows x columns) for non-square matrices, or simply the number of rows (or columns) for square
   matrices. The function iterates through the provided list and ensures that every item has the same dimension.

   .. rubric:: Examples

   Check a list of vectors with the same dimension:

   >>> import numpy as np
   >>> from toqito.matrix_props import has_same_dimension
   >>> vectors = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
   >>> has_same_dimension(vectors)
   True

   Check a list of matrices with the same dimension:

   >>> import numpy as np
   >>> from toqito.matrix_props import has_same_dimension
   >>> matrices = [np.array([[1, 0], [0, 1]]), np.array([[2, 3], [4, 5]]), np.array([[6, 7], [8, 9]])]
   >>> has_same_dimension(matrices)
   True

   Check a list containing items of different dimensions:

   >>> import numpy as np
   >>> from toqito.matrix_props import has_same_dimension
   >>> mixed = [np.array([1, 2, 3]), np.array([[1, 0], [0, 1]])]
   >>> has_same_dimension(mixed)
   False

   :param items: A list containing vectors or matrices. Vectors are represented as 1D numpy arrays, and matrices are
                 represented as 2D numpy arrays.
   :return: Returns :code:`True` if all items in the list have the same dimension, :code:`False` otherwise.
   :raises ValueError: If the input list is empty.



