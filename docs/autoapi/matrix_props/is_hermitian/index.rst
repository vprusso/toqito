matrix_props.is_hermitian
=========================

.. py:module:: matrix_props.is_hermitian

.. autoapi-nested-parse::

   Checks if the matrix is a Hermitian matrix.



Functions
---------

.. autoapisummary::

   matrix_props.is_hermitian.is_hermitian


Module Contents
---------------

.. py:function:: is_hermitian(mat, rtol = 1e-05, atol = 1e-08)

   Check if matrix is Hermitian :cite:`WikiHerm`.

   A Hermitian matrix is a complex square matrix that is equal to its own conjugate transpose.

   .. rubric:: Examples

   Consider the following matrix:

   .. math::
       A = \begin{pmatrix}
               2 & 2 +1j & 4 \\
               2 - 1j & 3 & 1j \\
               4 & -1j & 1
           \end{pmatrix}

   our function indicates that this is indeed a Hermitian matrix as it holds that

   .. math::
       A = A^*.

   >>> from toqito.matrix_props import is_hermitian
   >>> import numpy as np
   >>> mat = np.array([[2, 2 + 1j, 4], [2 - 1j, 3, 1j], [4, -1j, 1]])
   >>> is_hermitian(mat)
   True

   Alternatively, the following example matrix :math:`B` defined as

   .. math::
       B = \begin{pmatrix}
               1 & 2 & 3 \\
               4 & 5 & 6 \\
               7 & 8 & 9
           \end{pmatrix}

   is not Hermitian.

   >>> from toqito.matrix_props import is_hermitian
   >>> import numpy as np
   >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   >>> is_hermitian(mat)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: Matrix to check.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :param atol: The absolute tolerance parameter (default 1e-08).
   :return: Return True if matrix is Hermitian, and False otherwise.



