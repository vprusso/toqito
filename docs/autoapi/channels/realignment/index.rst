channels.realignment
====================

.. py:module:: channels.realignment

.. autoapi-nested-parse::

   Generates the realignment channel of a matrix.



Functions
---------

.. autoapisummary::

   channels.realignment.realignment


Module Contents
---------------

.. py:function:: realignment(input_mat, dim = None)

   Compute the realignment of a bipartite operator :cite:`Lupo_2008_Bipartite`.

   Gives the realignment of the matrix :code:`input_mat`, where it is assumed that the number
   of rows and columns of :code:`input_mat` are both perfect squares and both subsystems have
   equal dimension. The realignment is defined by mapping the operator :math:`|ij \rangle
   \langle kl |` to :math:`|ik \rangle \langle jl |` and extending linearly.

   If :code:`input_mat` is non-square, different row and column dimensions can be specified by
   putting the row dimensions in the first row of :code:`dim` and the column dimensions in the
   second row of :code:`dim`.

   .. rubric:: Examples

   The standard realignment map

   Using :code:`|toqito⟩`, we can generate the standard realignment map as follows. When viewed as a
   map on block matrices, the realignment map takes each block of the original matrix and makes
   its vectorization the rows of the realignment matrix. This is illustrated by the following
   small example:

   >>> from toqito.channels import realignment
   >>> import numpy as np
   >>> test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
   >>> realignment(test_input_mat)
   array([[ 1,  2,  5,  6],
          [ 3,  4,  7,  8],
          [ 9, 10, 13, 14],
          [11, 12, 15, 16]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param input_mat: The input matrix.
   :param dim: Default has all equal dimensions.
   :raises ValueError: If dimension of matrix is invalid.
   :return: The realignment map matrix.



