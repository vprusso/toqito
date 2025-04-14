matrix_props.is_unitary
=======================

.. py:module:: matrix_props.is_unitary

.. autoapi-nested-parse::

   Checks if the matrix is a unitary matrix.



Functions
---------

.. autoapisummary::

   matrix_props.is_unitary.is_unitary


Module Contents
---------------

.. py:function:: is_unitary(mat, rtol = 1e-05, atol = 1e-08)

   Check if matrix is unitary :cite:`WikiUniMat`.

   A matrix is unitary if its inverse is equal to its conjugate transpose.

   Alternatively, a complex square matrix :math:`U` is unitary if its conjugate transpose
   :math:`U^*` is also its inverse, that is, if

   .. math::
       \begin{equation}
           U^* U = U U^* = \mathbb{I},
       \end{equation}

   where :math:`\mathbb{I}` is the identity matrix.

   .. rubric:: Examples

   Consider the following matrix

   .. math::
       X = \begin{pmatrix}
           0 & 1 \\
           1 & 0
           \end{pmatrix}

   our function indicates that this is indeed a unitary matrix.

   >>> from toqito.matrix_props import is_unitary
   >>> import numpy as np
   >>> A = np.array([[0, 1], [1, 0]])
   >>> is_unitary(A)
   True

   We may also use the `random_unitary` function from `toqito`, and can verify that a randomly
   generated matrix is unitary

   >>> from toqito.matrix_props import is_unitary
   >>> from toqito.rand import random_unitary
   >>> mat = random_unitary(2)
   >>> is_unitary(mat)
   True

   Alternatively, the following example matrix :math:`B` defined as

   .. math::
       B = \begin{pmatrix}
           1 & 0 \\
           1 & 1
           \end{pmatrix}

   is not unitary.

   >>> from toqito.matrix_props import is_unitary
   >>> import numpy as np
   >>> B = np.array([[1, 0], [1, 1]])
   >>> is_unitary(B)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: Matrix to check.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :param atol: The absolute tolerance parameter (default 1e-08).
   :return: Return :code:`True` if matrix is unitary, and :code:`False` otherwise.



