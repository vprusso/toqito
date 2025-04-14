channel_props.is_quantum_channel
================================

.. py:module:: channel_props.is_quantum_channel

.. autoapi-nested-parse::

   Determines if an input is a quantum channel.



Functions
---------

.. autoapisummary::

   channel_props.is_quantum_channel.is_quantum_channel


Module Contents
---------------

.. py:function:: is_quantum_channel(phi, rtol = 1e-05, atol = 1e-08)

   Determine whether the given input is a quantum channel.

   For more info, see Section 2.2.1: Definitions and Basic Notions Concerning Channels from
   :cite:`Watrous_2018_TQI`.

   A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is a *quantum
   channel* for some choice of complex Euclidean spaces :math:`\mathcal{X}`
   and :math:`\mathcal{Y}`, if it holds that:

   1. :math:`\Phi` is completely positive.
   2. :math:`\Phi` is trace preserving.

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

   To check if this is a valid quantum channel or not,

   >>> import numpy as np
   >>> from toqito.matrices import pauli
   >>> from toqito.channel_props import is_quantum_channel
   >>> u = (1/np.sqrt(2))*np.array([[1, 1],[-1, 1]])
   >>> x = pauli("X")
   >>> phi = x - np.matmul(u, np.matmul(x, np.conjugate(u)))
   >>> is_quantum_channel(phi)
   False

   If we instead check for the validity of depolarizing channel being a valid quantum channel,

   >>> from toqito.channels import depolarizing
   >>> from toqito.channel_props import is_quantum_channel
   >>> choi_depolarizing = depolarizing(dim=2, param_p=0.2)
   >>> is_quantum_channel(choi_depolarizing)
   True

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :param atol: The absolute tolerance parameter (default 1e-08).
   :return: :code:`True` if the channel is a quantum channel, and :code:`False` otherwise.



