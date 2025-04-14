matrix_ops.calculate_vector_matrix_dimension
============================================

.. py:module:: matrix_ops.calculate_vector_matrix_dimension

.. autoapi-nested-parse::

   Calculates the (common) dimension of a set of vectors or matrices.



Functions
---------

.. autoapisummary::

   matrix_ops.calculate_vector_matrix_dimension.calculate_vector_matrix_dimension


Module Contents
---------------

.. py:function:: calculate_vector_matrix_dimension(item)

   Calculate the dimension of a vector or a square matrix, including 2D representations of vectors.

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

   >>> from toqito.matrix_ops import calculate_vector_matrix_dimension
   >>> import numpy as np
   >>> v = np.array([1, 0, 0])
   >>> calculate_vector_matrix_dimension(v)
   3

   For the density matrix of some two-dimensional quantum system

   .. math::
       \rho = \frac{1}{2}
               \begin{pmatrix}
                   1 & 0 \\
                   0 & 1
               \end{pmatrix}

   >>> from toqito.matrix_ops import calculate_vector_matrix_dimension
   >>> import numpy as np
   >>> rho = np.array([[1/2, 0],[0, 1/2]])
   >>> calculate_vector_matrix_dimension(rho)
   2



