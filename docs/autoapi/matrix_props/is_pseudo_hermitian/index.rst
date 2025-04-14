matrix_props.is_pseudo_hermitian
================================

.. py:module:: matrix_props.is_pseudo_hermitian

.. autoapi-nested-parse::

   Checks if matrix is pseudo hermitian with respect to given signature.



Functions
---------

.. autoapisummary::

   matrix_props.is_pseudo_hermitian.is_pseudo_hermitian


Module Contents
---------------

.. py:function:: is_pseudo_hermitian(mat, signature, rtol = 1e-05, atol = 1e-08)

   Check if a matrix is pseudo-Hermitian.

   A matrix :math:`H` is pseudo-Hermitian with respect to a given signature matrix :math:`\eta` if it satisfies:

   .. math::
       \eta H \eta^{-1} = H^{\dagger},

   where:
       - :math:`H^{\dagger}` is the conjugate transpose (Hermitian transpose) of :math:`H`,
       - :math:`\eta` is a Hermitian, invertible matrix.

   .. rubric:: Examples

   Consider the following matrix:

   .. math::
       H = \begin{pmatrix}
           1 & 1+i \\
           -1+i & -1
       \end{pmatrix}

   with the signature matrix:

   .. math::
       \eta = \begin{pmatrix}
           1 & 0 \\
           0 & -1
       \end{pmatrix}

   Our function confirms that :math:`H` is pseudo-Hermitian:

   >>> import numpy as np
   >>> from toqito.matrix_props import is_pseudo_hermitian
   >>> H = np.array([[1, 1+1j], [-1+1j, -1]])
   >>> eta = np.array([[1, 0], [0, -1]])
   >>> is_pseudo_hermitian(H, eta)
   True

   However, the following matrix :math:`A`

   .. math::
       A = \begin{pmatrix}
           1 & i \\
           -i & 1
       \end{pmatrix}

   is not pseudo-Hermitian with respect to the same signature matrix:

   >>> A = np.array([[1, 1j], [-1j, 1]])
   >>> is_pseudo_hermitian(A, eta)
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: The matrix to check.
   :param signature: The signature matrix :math:`\eta`, which must be Hermitian and invertible.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :param atol: The absolute tolerance parameter (default 1e-08).
   :raises ValueError: If `signature` is not Hermitian or not invertible.
   :return: Return :code:`True` if the matrix is pseudo-Hermitian, and :code:`False` otherwise.


