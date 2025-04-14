matrix_props.is_positive_definite
=================================

.. py:module:: matrix_props.is_positive_definite

.. autoapi-nested-parse::

   Checks if the matrix is a positive definite matrix.



Functions
---------

.. autoapisummary::

   matrix_props.is_positive_definite.is_positive_definite


Module Contents
---------------

.. py:function:: is_positive_definite(mat)

   Check if matrix is positive definite (PD) :cite:`WikiPosDef`.

   .. rubric:: Examples

   Consider the following matrix

   .. math::
       A = \begin{pmatrix}
               2 & -1 & 0 \\
               -1 & 2 & -1 \\
               0 & -1 & 2
           \end{pmatrix}

   our function indicates that this is indeed a positive definite matrix.

   >>> from toqito.matrix_props import is_positive_definite
   >>> import numpy as np
   >>> A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
   >>> is_positive_definite(A)
   True

   Alternatively, the following example matrix :math:`B` defined as

   .. math::
       B = \begin{pmatrix}
               -1 & -1 \\
               -1 & -1
           \end{pmatrix}

   is not positive definite.

   >>> from toqito.matrix_props import is_positive_definite
   >>> import numpy as np
   >>> B = np.array([[-1, -1], [-1, -1]])
   >>> is_positive_definite(B)
   False

   .. seealso:: :func:`.is_positive_semidefinite`

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: Matrix to check.
   :return: Return :code:`True` if matrix is positive definite, and :code:`False` otherwise.



