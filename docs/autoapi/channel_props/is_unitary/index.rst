channel_props.is_unitary
========================

.. py:module:: channel_props.is_unitary

.. autoapi-nested-parse::

   Determines if a channel is unitary.



Functions
---------

.. autoapisummary::

   channel_props.is_unitary.is_unitary


Module Contents
---------------

.. py:function:: is_unitary(phi)

   Given a quantum channel, determine if it is unitary.

   (Section 2.2.1: Definitions and Basic Notions Concerning Channels from
   :cite:`Watrous_2018_TQI`).

   Let :math:`\mathcal{X}` be a complex Euclidean space an let :math:`U \in U(\mathcal{X})` be a
   unitary operator. Then a unitary channel is defined as:

   .. math::
       \Phi(X) = U X U^*.

   .. rubric:: Examples

   The identity channel is one example of a unitary channel:

   .. math::
       U =
       \begin{pmatrix}
           1 & 0 \\
           0 & 1
       \end{pmatrix}.

   We can verify this as follows:

   >>> from toqito.channel_props import is_unitary
   >>> import numpy as np
   >>> kraus_ops = [[np.identity(2), np.identity(2)]]
   >>> is_unitary(kraus_ops)
   True

   We can also specify the input as a Choi matrix. For instance, consider the Choi matrix
   corresponding to the :math:`2`-dimensional completely depolarizing channel.

   .. math::
       \Omega =
       \frac{1}{2}
       \begin{pmatrix}
           1 & 0 & 0 & 0 \\
           0 & 1 & 0 & 0 \\
           0 & 0 & 1 & 0 \\
           0 & 0 & 0 & 1
       \end{pmatrix}.

   We may verify that this channel is not a unitary channel.

   >>> from toqito.channels import depolarizing
   >>> from toqito.channel_props import is_unitary
   >>> is_unitary(depolarizing(2))
   False

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
   :return: :code:`True` if the channel is a unitary channel, and :code:`False` otherwise.



