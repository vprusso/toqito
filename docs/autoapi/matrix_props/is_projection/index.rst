matrix_props.is_projection
==========================

.. py:module:: matrix_props.is_projection

.. autoapi-nested-parse::

   Checks if the matrix is a projection matrix.



Functions
---------

.. autoapisummary::

   matrix_props.is_projection.is_projection


Module Contents
---------------

.. py:function:: is_projection(mat, rtol = 1e-05, atol = 1e-08)

   Check if matrix is a projection matrix :cite:`WikiProjMat`.

   A matrix is a projection matrix if it is positive semidefinite (PSD) and if

   .. math::
       \begin{equation}
           X^2 = X
       \end{equation}

   where :math:`X` is the matrix in question.

   .. rubric:: Examples

   Consider the following matrix

   .. math::
       A = \begin{pmatrix}
               0 & 1 \\
               0 & 1
           \end{pmatrix}

   our function indicates that this is indeed a projection matrix.

   >>> from toqito.matrix_props import is_projection
   >>> import numpy as np
   >>> A = np.array([[0, 1], [0, 1]])
   >>> is_projection(A)
   True

   Alternatively, the following example matrix :math:`B` defined as

   .. math::
       B = \begin{pmatrix}
               -1 & -1 \\
               -1 & -1
           \end{pmatrix}

   is not positive definite.

   >>> from toqito.matrix_props import is_projection
   >>> import numpy as np
   >>> B = np.array([[-1, -1], [-1, -1]])
   >>> is_projection(B)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: Matrix to check.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :param atol: The absolute tolerance parameter (default 1e-08).
   :return: Return :code:`True` if matrix is a projection matrix, and :code:`False` otherwise.



