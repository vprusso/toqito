channel_props.is_trace_preserving
=================================

.. py:module:: channel_props.is_trace_preserving

.. autoapi-nested-parse::

   Determines if a channel is trace-preserving.



Functions
---------

.. autoapisummary::

   channel_props.is_trace_preserving.is_trace_preserving


Module Contents
---------------

.. py:function:: is_trace_preserving(phi, rtol = 1e-05, atol = 1e-08, sys = 2, dim = None)

   Determine whether the given channel is trace-preserving.

   A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is
   *trace-preserving* if it holds that

   .. math::
       \text{Tr} \left( \Phi(X) \right) = \text{Tr}\left( X \right)

   for every operator :math:`X \in \text{L}(\mathcal{X})`.

   Given the corresponding Choi matrix of the channel, a neccessary and sufficient condition is

   .. math::
       \text{Tr}_{\mathcal{Y}} \left( J(\Phi) \right) = \mathbb{I}_{\mathcal{X}}

   In case :code:`sys` is not specified, the default convention is that the Choi matrix
   is the result of applying the map to the second subsystem of the standard maximally
   entangled (unnormalized) state.

   The dimensions of the subsystems are given by the vector :code:`dim`. By default,
   both subsystems have equal dimension.

   Alternatively, given a list of Kraus operators, a neccessary and sufficient condition is

   .. math::
       \sum_{a \in \Sigma} A_a^* B_a = \mathbb{I}_{\mathcal{X}}

   .. rubric:: Examples

   The map :math:`\Phi` defined as

   .. math::
       \Phi(X) = X - U X U^*

   is not trace-preserving, where

   .. math::
       U = \frac{1}{\sqrt{2}}
       \begin{pmatrix}
           1 & 1 \\
           -1 & 1
       \end{pmatrix}.

   >>> import numpy as np
   >>> from toqito.channel_props import is_trace_preserving
   >>> unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
   >>> kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]
   >>> is_trace_preserving(kraus_ops)
   False

   As another example, the depolarizing channel is trace-preserving.

   >>> from toqito.channels import depolarizing
   >>> from toqito.channel_props import is_trace_preserving
   >>> choi_mat = depolarizing(2)
   >>> is_trace_preserving(choi_mat)
   True

   Further information for determining the trace preserving properties of channels consult (Section: Linear Maps Of
   Square Operators from :cite:`Watrous_2018_TQI`).

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
   :param rtol: The relative tolerance parameter (default 1e-05).
   :param atol: The absolute tolerance parameter (default 1e-08).
   :param sys: Scalar or vector specifying the size of the subsystems.
   :param dim: Dimension of the subsystems. If :code:`None`, all dimensions are assumed to be
               equal.
   :return: True if the channel is trace-preserving, and False otherwise.


