matrix_props.spark
==================

.. py:module:: matrix_props.spark

.. autoapi-nested-parse::

   Computes the spark of a matrix.



Functions
---------

.. autoapisummary::

   matrix_props.spark.spark


Module Contents
---------------

.. py:function:: spark(mat)

   Compute the spark of a matrix.

   The spark of a matrix A is the smallest number of columns from A that are linearly
   dependent :cite:`Elad_2010_Sparse`.

   .. rubric:: Examples

   >>> import numpy as np
   >>> from toqito.matrix_props import spark
   >>> A = np.array([[1, 0, 1, 2],
   ...               [0, 1, 1, 3],
   ...               [1, 1, 2, 5]])
   >>> spark(A)
   3

   .. rubric:: Notes

   - This function only works for 2D NumPy arrays.
   - If all columns are linearly independent, the function returns n_cols + 1.
   - The time complexity of this implementation is O(2^n) in the worst case,
     where n is the number of columns.
   - For an m x n matrix A with n >= m:
     * If spark(A) = m + 1, then rank(A) = m (full rank).
     * spark(A) = 1 if and only if the matrix has a zero column.
     * spark(A) <= rank(A) + 1.

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: The input matrix as a 2D NumPy array.
   :return: The spark of the input matrix :code:`mat`.
   :raises ValueError: If the input is not a 2D NumPy array.


