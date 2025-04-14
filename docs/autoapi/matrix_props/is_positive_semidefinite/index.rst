matrix_props.is_positive_semidefinite
=====================================

.. py:module:: matrix_props.is_positive_semidefinite

.. autoapi-nested-parse::

   Checks if the matrix is a positive semidefinite matrix.



Functions
---------

.. autoapisummary::

   matrix_props.is_positive_semidefinite.is_positive_semidefinite


Module Contents
---------------

.. py:function:: is_positive_semidefinite(mat, rtol = 1e-05, atol = 1e-08)

   Check if matrix is positive semidefinite (PSD) :cite:`WikiPosDef`.

   .. rubric:: Examples

   Consider the following matrix

   .. math::
       A = \begin{pmatrix}
               1 & -1 \\
               -1 & 1
           \end{pmatrix}

   our function indicates that this is indeed a positive semidefinite matrix.

   >>> from toqito.matrix_props import is_positive_semidefinite
   >>> import numpy as np
   >>> A = np.array([[1, -1], [-1, 1]])
   >>> is_positive_semidefinite(A)
   True

   Alternatively, the following example matrix :math:`B` defined as

   .. math::
       B = \begin{pmatrix}
               -1 & -1 \\
               -1 & -1
           \end{pmatrix}

   is not positive semidefinite.

   >>> from toqito.matrix_props import is_positive_semidefinite
   >>> import numpy as np
   >>> B = np.array([[-1, -1], [-1, -1]])
   >>> is_positive_semidefinite(B)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: Matrix to check.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :param atol: The absolute tolerance parameter (default 1e-08).
   :return: Return :code:`True` if matrix is PSD, and :code:`False` otherwise.



