matrix_props.positive_semidefinite_rank
=======================================

.. py:module:: matrix_props.positive_semidefinite_rank

.. autoapi-nested-parse::

   Calculates the positive semidefinite rank of a nonnegative matrix.



Functions
---------

.. autoapisummary::

   matrix_props.positive_semidefinite_rank.positive_semidefinite_rank
   matrix_props.positive_semidefinite_rank._check_psd_rank


Module Contents
---------------

.. py:function:: positive_semidefinite_rank(mat, max_rank = 10)

   Compute the positive semidefinite rank (PSD rank) of a nonnegative matrix.

   The definition of PSD rank is defined in :cite:`Fawzi_2015_Positive`.

   Finds the PSD rank of an input matrix by checking feasibility for increasing rank.

   .. rubric:: Examples

   As an example (Equation 21 from :cite:`Heinosaari_2024_Can`), the PSD rank of the following matrix

   .. math::
       A = \frac{1}{2}
       \begin{pmatrix}
           0 & 1 & 1 \\
           1 & 0 & 1 \\
           1 & 1 & 0
       \end{pmatrix}

   is known to be :math:`\text{rank}_{\text{PSD}}(A) = 2`.

   >>> import numpy as np
   >>> from toqito.matrix_props import positive_semidefinite_rank
   >>> positive_semidefinite_rank(1/2 * np.array([[0, 1, 1], [1,0,1], [1,1,0]]))
   2

   The PSD rank of the identity matrix is the dimension of the matrix :cite:`Fawzi_2015_Positive`.

   >>> import numpy as np
   >>> from toqito.matrix_props import positive_semidefinite_rank
   >>> positive_semidefinite_rank(np.identity(3))
   3

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param mat: 2D numpy ndarray.
   :param max_rank: The maximum rank to check.
   :return: The PSD rank of the input matrix, or None if not found within `max_rank`.


.. py:function:: _check_psd_rank(mat, max_rank)

   Check if the given PSD rank k is feasible for matrix M.

   :param mat: 2D numpy ndarray
   :param max_rank: The maximum rank to check.
   :return: True if `max_rank` is a feasible PSD rank, False otherwise.


