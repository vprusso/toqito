matrix_props.is_idempotent
==========================

.. py:module:: matrix_props.is_idempotent

.. autoapi-nested-parse::

   Checks if the matrix is an idempotent matrix.



Functions
---------

.. autoapisummary::

   matrix_props.is_idempotent.is_idempotent


Module Contents
---------------

.. py:function:: is_idempotent(mat, rtol = 1e-05, atol = 1e-08)

   Check if matrix is the idempotent matrix :cite:`WikiIdemPot`.

   An *idempotent matrix* is a square matrix, which, when multiplied by itself, yields itself.
   That is, the matrix :math:`A` is idempotent if and only if :math:`A^2 = A`.

   .. rubric:: Examples

   The following is an example of a :math:`2 x 2` idempotent matrix:

   .. math::
       A = \begin{pmatrix}
           3 & -6 \\
           1 & -2
       \end{pmatrix}

   >>> from toqito.matrix_props import is_idempotent
   >>> import numpy as np
   >>> mat = np.array([[3, -6], [1, -2]])
   >>> is_idempotent(mat)
   True

   Alternatively, the following matrix

   .. math::
       B = \begin{pmatrix}
               1 & 2 & 3 \\
               4 & 5 & 6 \\
               7 & 8 & 9
           \end{pmatrix}

   is not idempotent.

   >>> from toqito.matrix_props import is_idempotent
   >>> import numpy as np
   >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   >>> is_idempotent(mat)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: Matrix to check.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :param atol: The absolute tolerance parameter (default 1e-08).
   :return: Return :code:`True` if matrix is the idempotent matrix, and
           :code:`False` otherwise.



