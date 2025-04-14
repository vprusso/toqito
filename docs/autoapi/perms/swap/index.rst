perms.swap
==========

.. py:module:: perms.swap

.. autoapi-nested-parse::

   Swap is used to apply the swap function within a quantum state or an operator.



Functions
---------

.. autoapisummary::

   perms.swap.swap


Module Contents
---------------

.. py:function:: swap(rho, sys = None, dim = None, row_only = False)

   Swap two subsystems within a state or operator.

   Swaps the two subsystems of the vector or matrix :code:`rho`, where the dimensions of the (possibly more than 2)
   subsystems are given by :code:`dim` and the indices of the two subsystems to be swapped are specified in the 1-by-2
   vector :code:`sys`.

   If :code:`rho` is non-square and not a vector, different row and column dimensions can be specified by putting the
   row dimensions in the first row of :code:`dim` and the column dimensions in the second row of :code:`dim`.

   If :code:`row_only` is set to :code:`True`, then only the rows of :code:`rho` are swapped, but not the columns --
   this is equivalent to multiplying :code:`rho` on the left by the corresponding swap operator, but not on the right.

   .. rubric:: Examples

   Consider the following matrix

   .. math::
       X =
       \begin{pmatrix}
           1 & 5 & 9 & 13 \\
           2 & 6 & 10 & 14 \\
           3 & 7 & 11 & 15 \\
           4 & 8 & 12 & 16
       \end{pmatrix}.

   If we apply the :code:`swap` function provided by :code:`|toqito⟩` on :math:`X`, we should obtain the following
   matrix

   .. math::
       \text{Swap}(X) =
       \begin{pmatrix}
           1 & 9 & 5 & 13 \\
           3 & 11 & 7 & 15 \\
           2 & 10 & 6 & 14 \\
           4 & 12 & 8 & 16
       \end{pmatrix}.

   This can be observed by the following example in :code:`|toqito⟩`.

   >>> from toqito.perms import swap
   >>> import numpy as np
   >>> test_mat = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])
   >>> swap(test_mat)
   array([[ 1,  9,  5, 13],
          [ 3, 11,  7, 15],
          [ 2, 10,  6, 14],
          [ 4, 12,  8, 16]])

   It is also possible to use the :code:`sys` and :code:`dim` arguments, it is possible to specify the system and
   dimension on which to apply the swap operator. For instance for :code:`sys = [1 ,2]` and :code:`dim = 2` we have
   that

   .. math::
       \text{Swap}(X)_{2, [1, 2]} =
       \begin{pmatrix}
           1 & 9 & 5 & 13 \\
           3 & 11 & 7 & 15 \\
           2 & 10 & 6 & 14 \\
           4 & 12 & 8 & 16
       \end{pmatrix}.

   Using :code:`|toqito⟩` we can see this gives the proper result.

   >>> from toqito.perms import swap
   >>> import numpy as np
   >>> test_mat = np.array(
   ...     [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]]
   ... )
   >>> swap(test_mat, [1, 2], 2)
   array([[ 1,  9,  5, 13],
          [ 3, 11,  7, 15],
          [ 2, 10,  6, 14],
          [ 4, 12,  8, 16]])

   It is also possible to perform the :code:`swap` function on vectors in addition to matrices.

   >>> from toqito.perms import swap
   >>> import numpy as np
   >>> test_vec = np.array([1, 2, 3, 4])
   >>> swap(test_vec)
   array([1, 3, 2, 4])

   :raises ValueError: If dimension does not match the number of subsystems.
   :param rho: A vector or matrix to have its subsystems swapped.
   :param sys: Default: [1, 2]
   :param dim: Default: :code:`[sqrt(len(X), sqrt(len(X)))]`
   :param row_only: Default: :code:`False`
   :return: The swapped matrix.



