channel_ops.apply_channel
=========================

.. py:module:: channel_ops.apply_channel

.. autoapi-nested-parse::

   Applies a quantum channel to an operator.



Functions
---------

.. autoapisummary::

   channel_ops.apply_channel.apply_channel


Module Contents
---------------

.. py:function:: apply_channel(mat, phi_op)

   Apply a quantum channel to an operator.

   (Section: Representations and Characterizations of Channels of :cite:`Watrous_2018_TQI`).

   Specifically, an application of the channel is defined as

   .. math::
       \Phi(X) = \text{Tr}_{\mathcal{X}} \left(J(\Phi)
       \left(\mathbb{I}_{\mathcal{Y}} \otimes X^{T}\right)\right),

   where

   .. math::
       J(\Phi): \text{T}(\mathcal{X}, \mathcal{Y}) \rightarrow
       \text{L}(\mathcal{Y} \otimes \mathcal{X})

   is the Choi representation of :math:`\Phi`.

   We assume the quantum channel given as :code:`phi_op` is provided as either the Choi matrix
   of the channel or a set of Kraus operators that define the quantum channel.

   This function is adapted from the QETLAB package.

   .. rubric:: Examples

   The swap operator is the Choi matrix of the transpose map. The following is a (non-ideal,
   but illustrative) way of computing the transpose of a matrix.

   Consider the following matrix

   .. math::
       X = \begin{pmatrix}
               1 & 4 & 7 \\
               2 & 5 & 8 \\
               3 & 6 & 9
           \end{pmatrix}

   Applying the swap operator given as

   .. math::
       \Phi =
       \begin{pmatrix}
           1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
           0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
           0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
           0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
           0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
           0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
           0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
           0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
           0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{pmatrix}

   to the matrix :math:`X`, we have the resulting matrix of

   .. math::
       \Phi(X) = \begin{pmatrix}
                       1 & 2 & 3 \\
                       4 & 5 & 6 \\
                       7 & 8 & 9
                  \end{pmatrix}

   Using :code:`|toqito⟩`, we can obtain the above matrices as follows.

   >>> from toqito.channel_ops import apply_channel
   >>> from toqito.perms import swap_operator
   >>> import numpy as np
   >>> test_input_mat = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
   >>> apply_channel(test_input_mat, swap_operator(3))
   array([[1., 2., 3.],
          [4., 5., 6.],
          [7., 8., 9.]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If matrix is not Choi matrix.
   :param mat: A matrix.
   :param phi_op: A superoperator. :code:`phi_op` should be provided either as a Choi matrix,
                  or as a list of numpy arrays with either 1 or 2 columns whose entries are its
                  Kraus operators.
   :return: The result of applying the superoperator :code:`phi_op` to the operator :code:`mat`.



