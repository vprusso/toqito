perms.permute_systems
=====================

.. py:module:: perms.permute_systems

.. autoapi-nested-parse::

   Permute systems is used to permute subsystems within a quantum state or an operator.



Functions
---------

.. autoapisummary::

   perms.permute_systems.permute_systems


Module Contents
---------------

.. py:function:: permute_systems(input_mat, perm, dim = None, row_only = False, inv_perm = False)

   Permute subsystems within a state or operator.

   Permutes the order of the subsystems of the vector or matrix :code:`input_mat` according to the permutation vector
   :code:`perm`, where the dimensions of the subsystems are given by the vector :code:`dim`. If :code:`input_mat` is
   non-square and not a vector, different row and column dimensions can be specified by putting the row dimensions in
   the first row of :code:`dim` and the columns dimensions in the second row of :code:`dim`.

   If :code:`row_only = True`, then only the rows of :code:`input_mat` are permuted, but not the columns -- this is
   equivalent to multiplying :code:`input_mat` on the left by the corresponding permutation operator, but not on the
   right.

   If :code:`row_only = False`, then :code:`dim` only needs to contain the row dimensions of the subsystems, even if
   :code:`input_mat` is not square. If :code:`inv_perm = True`, then the inverse permutation of :code:`perm` is applied
   instead of :code:`perm` itself.

   .. rubric:: Examples

   For spaces :math:`\mathcal{A}` and :math:`\mathcal{B}` where :math:`\text{dim}(\mathcal{A}) =
   \text{dim}(\mathcal{B}) = 2` we may consider an operator :math:`X \in \mathcal{A} \otimes \mathcal{B}`. Applying the
   `permute_systems` function with vector :math:`[1,0]` on :math:`X`, we may reorient the spaces such that :math:`X \in
   \mathcal{B} \otimes \mathcal{A}`.

   For example, if we define :math:`X \in \mathcal{A} \otimes \mathcal{B}` as

   .. math::
       X = \begin{pmatrix}
           1 & 2 & 3 & 4 \\
           5 & 6 & 7 & 8 \\
           9 & 10 & 11 & 12 \\
           13 & 14 & 15 & 16
       \end{pmatrix},

   then applying the `permute_systems` function on :math:`X` to obtain :math:`X \in \mathcal{B} \otimes \mathcal{A}`
   yield the following matrix

   .. math::
       X_{[1,0]} = \begin{pmatrix}
           1 & 3 & 2 & 4 \\
           9 & 11 & 10 & 12 \\
           5 & 7 & 6 & 8 \\
           13 & 15 & 14 & 16
       \end{pmatrix}.

   >>> from toqito.perms import permute_systems
   >>> import numpy as np
   >>> test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
   >>> permute_systems(test_input_mat, [1, 0])
   array([[ 1,  3,  2,  4],
          [ 9, 11, 10, 12],
          [ 5,  7,  6,  8],
          [13, 15, 14, 16]])

   For spaces :math:`\mathcal{A}, \mathcal{B}`, and :math:`\mathcal{C}` where :math:`\text{dim}(\mathcal{A}) =
   \text{dim}(\mathcal{B}) = \text{dim}(\mathcal{C}) = 2` we may consider an operator :math:`X \in \mathcal{A} \otimes
   \mathcal{B} \otimes \mathcal{C}`. Applying the :code:`permute_systems` function with vector :math:`[1,2,0]` on
   :math:`X`, we may reorient the spaces such that :math:`X \in \mathcal{B} \otimes \mathcal{C} \otimes \mathcal{A}`.

   For example, if we define :math:`X \in \mathcal{A} \otimes \mathcal{B} \otimes \mathcal{C}` as

   .. math::
       X =
       \begin{pmatrix}
           1 & 2 & 3 & 4, 5 & 6 & 7 & 8 \\
           9 & 10 & 11 & 12 & 13 & 14 & 15 & 16 \\
           17 & 18 & 19 & 20 & 21 & 22 & 23 & 24 \\
           25 & 26 & 27 & 28 & 29 & 30 & 31 & 32 \\
           33 & 34 & 35 & 36 & 37 & 38 & 39 & 40 \\
           41 & 42 & 43 & 44 & 45 & 46 & 47 & 48 \\
           49 & 50 & 51 & 52 & 53 & 54 & 55 & 56 \\
           57 & 58 & 59 & 60 & 61 & 62 & 63 & 64
       \end{pmatrix},

   then applying the `permute_systems` function on :math:`X` to obtain :math:`X \in \mathcal{B} \otimes \mathcal{C}
   \otimes \mathcal{C}` yield the following matrix

   .. math::
       X_{[1, 2, 0]} =
       \begin{pmatrix}
           1 & 5 & 2 & 6 & 3 & 7 & 4, 8 \\
           33 & 37 & 34 & 38 & 35 & 39 & 36 & 40 \\
           9 & 13 & 10 & 14 & 11 & 15 & 12 & 16 \\
           41 & 45 & 42 & 46 & 43 & 47 & 44 & 48 \\
           17 & 21 & 18 & 22 & 19 & 23 & 20 & 24 \\
           49 & 53 & 50 & 54 & 51 & 55 & 52 & 56 \\
           25 & 29 & 26 & 30 & 27 & 31 & 28 & 32 \\
           57 & 61 & 58 & 62 & 59 & 63 & 60 & 64
       \end{pmatrix}.

   >>> from toqito.perms import permute_systems
   >>> import numpy as np
   >>> test_input_mat = np.array(
   ...    [
   ...        [1, 2, 3, 4, 5, 6, 7, 8],
   ...        [9, 10, 11, 12, 13, 14, 15, 16],
   ...        [17, 18, 19, 20, 21, 22, 23, 24],
   ...        [25, 26, 27, 28, 29, 30, 31, 32],
   ...        [33, 34, 35, 36, 37, 38, 39, 40],
   ...        [41, 42, 43, 44, 45, 46, 47, 48],
   ...        [49, 50, 51, 52, 53, 54, 55, 56],
   ...        [57, 58, 59, 60, 61, 62, 63, 64],
   ...    ]
   ... )
   >>> permute_systems(test_input_mat, [1, 2, 0])
   array([[ 1,  5,  2,  6,  3,  7,  4,  8],
          [33, 37, 34, 38, 35, 39, 36, 40],
          [ 9, 13, 10, 14, 11, 15, 12, 16],
          [41, 45, 42, 46, 43, 47, 44, 48],
          [17, 21, 18, 22, 19, 23, 20, 24],
          [49, 53, 50, 54, 51, 55, 52, 56],
          [25, 29, 26, 30, 27, 31, 28, 32],
          [57, 61, 58, 62, 59, 63, 60, 64]])

   :raises ValueError: If dimension does not match the number of subsystems.
   :param input_mat: The vector or matrix.
   :param perm: A permutation vector.
   :param dim: The default has all subsystems of equal dimension.
   :param row_only: Default: :code:`False`
   :param inv_perm: Default: :code:`True`
   :return: The matrix or vector that has been permuted.



