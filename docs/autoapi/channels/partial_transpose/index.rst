channels.partial_transpose
==========================

.. py:module:: channels.partial_transpose

.. autoapi-nested-parse::

   Generates the partial transpose of a matrix.



Functions
---------

.. autoapisummary::

   channels.partial_transpose.partial_transpose


Module Contents
---------------

.. py:function:: partial_transpose(rho, sys = None, dim = None)

   Compute the partial transpose of a matrix :cite:`WikiPeresHorodecki`.

   The *partial transpose* is defined as

   .. math::
       \left( \text{T} \otimes \mathbb{I}_{\mathcal{Y}} \right)
       \left(X\right)

   where :math:`X \in \text{L}(\mathcal{X})` is a linear operator over the complex Euclidean
   space :math:`\mathcal{X}` and where :math:`\text{T}` is the transpose mapping
   :math:`\text{T} \in \text{T}(\mathcal{X})` defined as

   .. math::
       \text{T}(X) = X^{\text{T}}

   for all :math:`X \in \text{L}(\mathcal{X})`.

   By default, the returned matrix is the partial transpose of the matrix :code:`rho`, where it
   is assumed that the number of rows and columns of :code:`rho` are both perfect squares and
   both subsystems have equal dimension. The transpose is applied to the second subsystem.

   In the case where :code:`sys` amd :code:`dim` are specified, this function gives the partial
   transpose of the matrix :code:`rho` where the dimensions of the (possibly more than 2)
   subsystems are given by the vector :code:`dim` and the subsystems to take the partial
   transpose are given by the scalar or vector :code:`sys`. If :code:`rho` is non-square,
   different row and column dimensions can be specified by putting the row dimensions in the
   first row of :code:`dim` and the column dimensions in the second row of :code:`dim`.

   .. rubric:: Examples

   Consider the following matrix

   .. math::
       X = \begin{pmatrix}
               1 & 2 & 3 & 4 \\
               5 & 6 & 7 & 8 \\
               9 & 10 & 11 & 12 \\
               13 & 14 & 15 & 16
           \end{pmatrix}.

   Performing the partial transpose on the matrix :math:`X` over the second
   subsystem yields the following matrix

   .. math::
       X_{pt, 2} = \begin{pmatrix}
                   1 & 5 & 3 & 7 \\
                   2 & 6 & 4 & 8 \\
                   9 & 13 & 11 & 15 \\
                   10 & 14 & 12 & 16
                \end{pmatrix}.

   By default, in :code:`|toqito⟩`, the partial transpose function performs the transposition on
   the second subsystem as follows.

   >>> from toqito.channels import partial_transpose
   >>> import numpy as np
   >>> test_input_mat = np.arange(1, 17).reshape(4, 4)
   >>> partial_transpose(test_input_mat)
   array([[ 1,  5,  3,  7],
          [ 2,  6,  4,  8],
          [ 9, 13, 11, 15],
          [10, 14, 12, 16]])

   By specifying the :code:`sys = 1` argument, we can perform the partial transpose over the
   first subsystem (instead of the default second subsystem as done above). Performing the
   partial transpose over the first subsystem yields the following matrix

   .. math::
       X_{pt, 1} = \begin{pmatrix}
                       1 & 2 & 9 & 10 \\
                       5 & 6 & 13 & 14 \\
                       3 & 4 & 11 & 12 \\
                       7 & 8 & 15 & 16
                   \end{pmatrix}.

   >>> from toqito.channels import partial_transpose
   >>> import numpy as np
   >>> test_input_mat = np.array(
   ...     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
   ... )
   >>> partial_transpose(test_input_mat, 1)
   array([[ 1,  5,  3,  7],
          [ 2,  6,  4,  8],
          [ 9, 13, 11, 15],
          [10, 14, 12, 16]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param rho: A matrix.
   :param sys: Scalar or vector specifying the size of the subsystems.
   :param dim: Dimension of the subsystems. If :code:`None`, all dimensions
               are assumed to be equal.
   :raises ValueError: If matrix dimensions are not square.
   :returns: The partial transpose of matrix :code:`rho`.



