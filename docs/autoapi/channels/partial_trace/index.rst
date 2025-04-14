channels.partial_trace
======================

.. py:module:: channels.partial_trace

.. autoapi-nested-parse::

   Generates the partial trace of a matrix.



Functions
---------

.. autoapisummary::

   channels.partial_trace.partial_trace


Module Contents
---------------

.. py:function:: partial_trace(input_mat, sys = None, dim = None)

   Compute the partial trace of a matrix :cite:`WikiPartialTr`.

   The *partial trace* is defined as

   .. math::
       \left( \text{Tr} \otimes \mathbb{I}_{\mathcal{Y}} \right)
       \left(X \otimes Y \right) = \text{Tr}(X)Y

   where :math:`X \in \text{L}(\mathcal{X})` and :math:`Y \in \text{L}(\mathcal{Y})` are linear
   operators over complex Euclidean spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}`.

   Gives the partial trace of the matrix X, where the dimensions of the (possibly more than 2)
   subsystems are given by the vector :code:`dim` and the subsystems to take the trace on are
   given by the scalar or vector :code:`sys`.

   .. rubric:: Examples

   Consider the following matrix

   .. math::
       X = \begin{pmatrix}
               1 & 2 & 3 & 4 \\
               5 & 6 & 7 & 8 \\
               9 & 10 & 11 & 12 \\
               13 & 14 & 15 & 16
           \end{pmatrix}.

   Taking the partial trace over the second subsystem of :math:`X` yields the following matrix

   .. math::
       X_{pt, 2} = \begin{pmatrix}
                   7 & 11 \\
                   23 & 27
                \end{pmatrix}.

   By default, the partial trace function in :code:`|toqito⟩` takes the trace of the second
   subsystem.

   >>> from toqito.channels import partial_trace
   >>> import numpy as np
   >>> test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
   >>> partial_trace(test_input_mat)
   array([[ 7, 11],
          [23, 27]])

   By specifying the :code:`sys = [0]` argument, we can perform the partial trace over the first
   subsystem (instead of the default second subsystem as done above). Performing the partial
   trace over the first subsystem yields the following matrix

   .. math::
       X_{pt, 1} = \begin{pmatrix}
                       12 & 14 \\
                       20 & 22
                   \end{pmatrix}

   >>> from toqito.channels import partial_trace
   >>> import numpy as np
   >>> test_input_mat = np.array(
   ...     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
   ... )
   >>> partial_trace(test_input_mat, [0])
   array([[12, 14],
          [20, 22]])

   We can also specify both dimension and system size as :code:`list` arguments. Consider the
   following :math:`16`-by-:math:`16` matrix.

   >>> from toqito.channels import partial_trace
   >>> import numpy as np
   >>> test_input_mat = np.arange(1, 257).reshape(16, 16)
   >>> test_input_mat
   array([[  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
            14,  15,  16],
          [ 17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
            30,  31,  32],
          [ 33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
            46,  47,  48],
          [ 49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
            62,  63,  64],
          [ 65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
            78,  79,  80],
          [ 81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,
            94,  95,  96],
          [ 97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
           110, 111, 112],
          [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
           126, 127, 128],
          [129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
           142, 143, 144],
          [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
           158, 159, 160],
          [161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
           174, 175, 176],
          [177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
           190, 191, 192],
          [193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
           206, 207, 208],
          [209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221,
           222, 223, 224],
          [225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
           238, 239, 240],
          [241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
           254, 255, 256]])

   We can take the partial trace on the first and third subsystems and assume that the size of
   each of the 4 systems is of dimension 2.

   >>> from toqito.channels import partial_trace
   >>> import numpy as np
   >>> partial_trace(test_input_mat, [0, 2], [2, 2, 2, 2])
   array([[344, 348, 360, 364],
          [408, 412, 424, 428],
          [600, 604, 616, 620],
          [664, 668, 680, 684]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If matrix dimension is not equal to the number of subsystems.
   :param input_mat: A square matrix.
   :param sys: Scalar or vector specifying the size of the subsystems.
   :param dim: Dimension of the subsystems. If :code:`None`, all dimensions are assumed to be
               equal.
   :return: The partial trace of matrix :code:`input_mat`.



