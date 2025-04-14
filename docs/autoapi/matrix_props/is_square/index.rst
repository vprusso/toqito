matrix_props.is_square
======================

.. py:module:: matrix_props.is_square

.. autoapi-nested-parse::

   Checks if the matrix is a square matrix.



Functions
---------

.. autoapisummary::

   matrix_props.is_square.is_square


Module Contents
---------------

.. py:function:: is_square(mat)

   Determine if a matrix is square :cite:`WikiSqMat`.

   A matrix is square if the dimensions of the rows and columns are equivalent.

   .. rubric:: Examples

   Consider the following matrix

   .. math::
       A = \begin{pmatrix}
               1 & 2 & 3 \\
               4 & 5 & 6 \\
               7 & 8 & 9
           \end{pmatrix}

   our function indicates that this is indeed a square matrix.

   >>> from toqito.matrix_props import is_square
   >>> import numpy as np
   >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   >>> is_square(A)
   True

   Alternatively, the following example matrix :math:`B` defined as

   .. math::
       B = \begin{pmatrix}
               1 & 2 & 3 \\
               4 & 5 & 6
           \end{pmatrix}

   is not square.

   >>> from toqito.matrix_props import is_square
   >>> import numpy as np
   >>> B = np.array([[1, 2, 3], [4, 5, 6]])
   >>> is_square(B)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If variable is not a matrix.
   :param mat: The matrix to check.
   :return: Returns :code:`True` if the matrix is square and :code:`False` otherwise.



