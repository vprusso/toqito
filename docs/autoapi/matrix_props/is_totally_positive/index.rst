matrix_props.is_totally_positive
================================

.. py:module:: matrix_props.is_totally_positive

.. autoapi-nested-parse::

   Checks if the matrix is totally positive.



Functions
---------

.. autoapisummary::

   matrix_props.is_totally_positive.is_totally_positive


Module Contents
---------------

.. py:function:: is_totally_positive(mat, tol = 1e-06, sub_sizes = None)

   Determine whether a matrix is totally positive. :cite:`WikiTotPosMat`.

   A totally positive matrix is a square matrix where all the minors are positive. Equivalently, the determinant of
   every square submatrix is a positive number.

   .. rubric:: Examples

   Consider the matrix

   .. math::
       \begin{pmatrix}
           1 & 2 \\
           3 & 4
       \end{pmatrix}

   To determine if this matrix is totally positive, we need to check the positivity of all of its minors. The 1x1
   minors are simply the individual entries of the matrix. For :math:`X`, these are

   .. math::
       \begin{equation}
           \begin{aligned}
               X_{1,1} &= 1 \\
               X_{1,2} &= 2 \\
               X_{2,1} &= 3 \\
               X_{2,2} &= 4 \\
           \end{aligned}
       \end{equation}

   Each of these entries is positive. There is only one 2x2 minor in this case, which is the determinant of the entire
   matrix :math:`X`. The determinant of :math:`X` is calculated as:

   .. math::
       \text{det}(X) = 1 \times 4 - 2 \times 3 = 4 - 6 = 2

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: Matrix to check.
   :param tol: The absolute tolerance parameter (default 1e-06).
   :param sub_sizes: List of sizes of submatrices to consider. Default is all sizes up to :code:`min(mat.shape)`.
   :return: Return :code:`True` if matrix is totally positive, and :code:`False` otherwise.



