matrix_props.is_diagonally_dominant
===================================

.. py:module:: matrix_props.is_diagonally_dominant

.. autoapi-nested-parse::

   Checks if the matrix is diagonally dominant.



Functions
---------

.. autoapisummary::

   matrix_props.is_diagonally_dominant.is_diagonally_dominant


Module Contents
---------------

.. py:function:: is_diagonally_dominant(mat, is_strict = True)

   Check if matrix is diagnal dominant (DD) :cite:`WikiDiagDom`.

   A matrix is diagonally dominant if the matrix is square
   and if for every row of the matrix, the magnitude of the diagonal entry in a row is greater
   than or equal to the sum of the magnitudes of all the other (non-diagonal) entries in that row.

   .. rubric:: Examples

   The following is an example of a 3-by-3 diagonal matrix:

      .. math::
          A = \begin{pmatrix}
                  2 & -1 & 0 \\
                  0 & 2 & -1 \\
                  0 & -1 & 2
              \end{pmatrix}

   our function indicates that this is indeed a diagonally dominant matrix.

   >>> from toqito.matrix_props import is_diagonally_dominant
   >>> import numpy as np
   >>> A = np.array([[2, -1, 0], [0, 2, -1], [0, -1, 2]])
   >>> is_diagonally_dominant(A)
   np.True_

   Alternatively, the following example matrix :math:`B` defined as

      .. math::
          B = \begin{pmatrix}
                  -1 & 2 \\
                  -1 & -1
              \end{pmatrix}

   is not diagonally dominant.

   >>> from toqito.matrix_props import is_diagonally_dominant
   >>> import numpy as np
   >>> B = np.array([[-1, 2], [-1, -1]])
   >>> is_diagonally_dominant(B)
   np.False_

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: Matrix to check.
   :param is_strict: Whether the inequality is strict.
   :return: Return :code:`True` if matrix is diagnally dominant, and :code:`False` otherwise.



