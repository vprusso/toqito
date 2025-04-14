matrices.cyclic_permutation_matrix
==================================

.. py:module:: matrices.cyclic_permutation_matrix

.. autoapi-nested-parse::

   Generates a cyclic permutation matrix.



Functions
---------

.. autoapisummary::

   matrices.cyclic_permutation_matrix.cyclic_permutation_matrix


Module Contents
---------------

.. py:function:: cyclic_permutation_matrix(n, k = 1)

   Create the cyclic permutation matrix for a given dimension :code:`n` :cite:`WikiCyclicPermutation`.

   This function creates a cyclic permutation matrix of 0's and 1's which is a special type of square matrix
   that represents a cyclic permutation of its rows. The function allows fixed points and successive applications.

   .. rubric:: Examples

   Generate fixed point.

   >>> from toqito.matrices import cyclic_permutation_matrix
   >>> n = 4
   >>> cyclic_permutation_matrix(n)
   array([[0, 0, 0, 1],
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0]])

   Generate successive application.

   >>> from toqito.matrices import cyclic_permutation_matrix
   >>> n = 4
   >>> k = 3
   >>> cyclic_permutation_matrix(n, k)
   array([[0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
          [1, 0, 0, 0]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param n: int
       The number of rows and columns in the cyclic permutation matrix.

   :param k: int
       The power to which the elements are raised, representing successive applications.

   :return:
       A NumPy array representing a cyclic permutation matrix of dimension :code:`n x n`.
       Each row of the matrix is shifted one position to the right in a cyclic manner,
       creating a circular permutation pattern. If :code:`k` is specified, the function
       raises the matrix to the power of :code:`k`, representing successive applications
       of the cyclic permutation.


