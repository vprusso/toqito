matrix_ops.to_density_matrix
============================

.. py:module:: matrix_ops.to_density_matrix

.. autoapi-nested-parse::

   Converts a row or a column vector to a density matrix.



Functions
---------

.. autoapisummary::

   matrix_ops.to_density_matrix.to_density_matrix


Module Contents
---------------

.. py:function:: to_density_matrix(input_array)

   Convert a given vector to a density matrix or return the density matrix if already given.

   If the input is a vector, this function computes the outer product to form a density matrix.
   If the input is already a density matrix (square matrix), it returns the matrix as is.

   .. rubric:: Examples

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



