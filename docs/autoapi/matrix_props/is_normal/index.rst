matrix_props.is_normal
======================

.. py:module:: matrix_props.is_normal

.. autoapi-nested-parse::

   Checks if the matrix is a normal matrix.



Functions
---------

.. autoapisummary::

   matrix_props.is_normal.is_normal


Module Contents
---------------

.. py:function:: is_normal(mat, rtol = 1e-05, atol = 1e-08)

   Determine if a matrix is normal :cite:`WikiNorm`.

   A matrix is normal if it commutes with its adjoint

   .. math::
       \begin{equation}
           [X, X^*] = 0,
       \end{equation}

   or, equivalently if

   .. math::
       \begin{equation}
           X^* X = X X^*
       \end{equation}.

   .. rubric:: Examples

   Consider the following matrix

   .. math::
       A = \begin{pmatrix}
               1 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 \\
               0 & 0 & 1 & 0 \\
               0 & 0 & 0 & 1
           \end{pmatrix}

   our function indicates that this is indeed a normal matrix.

   >>> from toqito.matrix_props import is_normal
   >>> import numpy as np
   >>> A = np.identity(4)
   >>> is_normal(A)
   True

   Alternatively, the following example matrix :math:`B` defined as

   .. math::
       B = \begin{pmatrix}
               1 & 2 & 3 \\
               4 & 5 & 6 \\
               7 & 8 & 9
           \end{pmatrix}

   is not normal.

   >>> from toqito.matrix_props import is_normal
   >>> import numpy as np
   >>> B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   >>> is_normal(B)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: The matrix to check.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :param atol: The absolute tolerance parameter (default 1e-08).
   :return: Returns :code:`True` if the matrix is normal and :code:`False` otherwise.



