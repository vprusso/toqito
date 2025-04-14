channel_props.is_herm_preserving
================================

.. py:module:: channel_props.is_herm_preserving

.. autoapi-nested-parse::

   Determines if a channel is Hermiticity-preserving.



Functions
---------

.. autoapisummary::

   channel_props.is_herm_preserving.is_herm_preserving


Module Contents
---------------

.. py:function:: is_herm_preserving(phi, rtol = 1e-05, atol = 1e-08)

   Determine whether the given channel is Hermitian-preserving.

   (Section: Linear Maps Of Square Operators from :cite:`Watrous_2018_TQI`).

   A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is
   *Hermitian-preserving* if it holds that

   .. math::
       \Phi(H) \in \text{Herm}(\mathcal{Y})

   for every Hermitian operator :math:`H \in \text{Herm}(\mathcal{X})`.

   .. rubric:: Examples

   The map :math:`\Phi` defined as

   .. math::
       \Phi(X) = X - U X U^*

   is Hermitian-preserving, where

   .. math::
       U = \frac{1}{\sqrt{2}}
       \begin{pmatrix}
           1 & 1 \\
           -1 & 1
       \end{pmatrix}.

   >>> import numpy as np
   >>> from toqito.channel_props import is_herm_preserving
   >>> unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
   >>> kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]
   >>> is_herm_preserving(kraus_ops)
   True

   We may also verify whether the corresponding Choi matrix of a given map is
   Hermitian-preserving. The swap operator is the Choi matrix of the transpose map, which is
   Hermitian-preserving as can be seen as follows:

   >>> import numpy as np
   >>> from toqito.perms import swap_operator
   >>> from toqito.channel_props import is_herm_preserving
   >>> unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
   >>> choi_mat = swap_operator(3)
   >>> is_herm_preserving(choi_mat)
   True

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :param atol: The absolute tolerance parameter (default 1e-08).
   :return: True if the channel is Hermitian-preserving, and False otherwise.



