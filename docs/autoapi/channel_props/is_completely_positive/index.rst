channel_props.is_completely_positive
====================================

.. py:module:: channel_props.is_completely_positive

.. autoapi-nested-parse::

   Determines if a channel is completely positive.



Functions
---------

.. autoapisummary::

   channel_props.is_completely_positive.is_completely_positive


Module Contents
---------------

.. py:function:: is_completely_positive(phi, rtol = 1e-05, atol = 1e-08)

   Determine whether the given channel is completely positive.

   (Section: Linear Maps Of Square Operators from :cite:`Watrous_2018_TQI`).

   A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is *completely
   positive* if it holds that

   .. math::
       \Phi \otimes \mathbb{I}_{\text{L}(\mathcal{Z})}

   is a positive map for every complex Euclidean space :math:`\mathcal{Z}`.

   Alternatively, a channel is completely positive if the corresponding Choi matrix of the
   channel is both Hermitian-preserving and positive semidefinite.

   .. rubric:: Examples

   We can specify the input as a list of Kraus operators. Consider the map :math:`\Phi` defined as

   .. math::
       \Phi(X) = X - U X U^*

   where

   .. math::
       U = \frac{1}{\sqrt{2}}
       \begin{pmatrix}
           1 & 1 \\
           -1 & 1
       \end{pmatrix}.

   This map is not completely positive, as we can verify as follows.

   >>> from toqito.channel_props import is_completely_positive
   >>> import numpy as np
   >>> unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
   >>> kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]
   >>> is_completely_positive(kraus_ops)
   False

   We can also specify the input as a Choi matrix. For instance, consider the Choi matrix
   corresponding to the :math:`2`-dimensional completely depolarizing channel

   .. math::
       \Omega =
       \frac{1}{2}
       \begin{pmatrix}
           1 & 0 & 0 & 0 \\
           0 & 1 & 0 & 0 \\
           0 & 0 & 1 & 0 \\
           0 & 0 & 0 & 1
       \end{pmatrix}.

   We may verify that this channel is completely positive

   >>> from toqito.channels import depolarizing
   >>> from toqito.channel_props import is_completely_positive
   >>> is_completely_positive(depolarizing(2))
   True

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :param atol: The absolute tolerance parameter (default 1e-08).
   :return: True if the channel is completely positive, and False otherwise.



