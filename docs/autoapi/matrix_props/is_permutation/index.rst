matrix_props.is_permutation
===========================

.. py:module:: matrix_props.is_permutation

.. autoapi-nested-parse::

   Checks if the matrix is a permutation matrix.



Functions
---------

.. autoapisummary::

   matrix_props.is_permutation.is_permutation


Module Contents
---------------

.. py:function:: is_permutation(mat)

   Determine if a matrix is a permutation matrix :cite:`WikiPerm`.

   A matrix is a permutation matrix if each row and column has a
   single element of 1 and all others are 0.

   .. rubric:: Examples

   Consider the following permutation matrix

   .. math::
       A = \begin{pmatrix}
               1 & 0 & 0 \\
               0 & 0 & 1 \\
               0 & 1 & 0
           \end{pmatrix}

   which is indeed a permutation matrix.

   >>> from toqito.matrix_props import is_permutation
   >>> import numpy as np
   >>> A = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
   >>> is_permutation(A)
   True

   Alternatively, the following example matrix :math:`B` defined as

   .. math::
       B = \begin{pmatrix}
               1 & 0 & 0 \\
               1 & 0 & 0 \\
               1 & 0 & 0
           \end{pmatrix}

   has 2 columns with all zero values and is thus not a
   permutation matrix.

   >>> from toqito.matrix_props import is_permutation
   >>> import numpy as np
   >>> B = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
   >>> is_permutation(B)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: The matrix to check.
   :return: Returns :code:`True` if the matrix is a permutation matrix and :code:`False` otherwise.



