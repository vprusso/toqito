channel_ops.partial_channel
===========================

.. py:module:: channel_ops.partial_channel

.. autoapi-nested-parse::

   Applies a channel to a subsystem of an operator.



Functions
---------

.. autoapisummary::

   channel_ops.partial_channel.partial_channel


Module Contents
---------------

.. py:function:: partial_channel(rho, phi_map, sys = 2, dim = None)

   Apply channel to a subsystem of an operator :cite:`Watrous_2018_TQI`.

   Applies the operator

   .. math::
       \left(\mathbb{I} \otimes \Phi \right) \left(\rho \right).

   In other words, it is the result of applying the channel :math:`\Phi` to the second subsystem
   of :math:`\rho`, which is assumed to act on two subsystems of equal dimension.

   The input :code:`phi_map` should be provided as a Choi matrix.

   This function is adapted from the QETLAB package.

   .. rubric:: Examples

   The following applies the completely depolarizing channel to the second
   subsystem of a random density matrix.

   >>> import numpy as np
   >>> from toqito.channel_ops import partial_channel
   >>> from toqito.channels import depolarizing
   >>> rho = np.array([
   ...    [0.3101, -0.0220-0.0219*1j, -0.0671-0.0030*1j, -0.0170-0.0694*1j],
   ...    [-0.0220+0.0219*1j, 0.1008, -0.0775+0.0492*1j, -0.0613+0.0529*1j],
   ...    [-0.0671+0.0030*1j, -0.0775-0.0492*1j, 0.1361, 0.0602 + 0.0062*1j],
   ...    [-0.0170+0.0694*1j, -0.0613-0.0529*1j, 0.0602-0.0062*1j, 0.4530]])
   >>> partial_channel(rho, depolarizing(2))
   array([[ 0.20545+0.j     ,  0.     +0.j     , -0.0642 +0.02495j,
            0.     +0.j     ],
          [ 0.     +0.j     ,  0.20545+0.j     ,  0.     +0.j     ,
           -0.0642 +0.02495j],
          [-0.0642 -0.02495j,  0.     +0.j     ,  0.29455+0.j     ,
            0.     +0.j     ],
          [ 0.     +0.j     , -0.0642 -0.02495j,  0.     +0.j     ,
            0.29455+0.j     ]])

   The following applies the completely depolarizing channel to the first
   subsystem.

   >>> import numpy as np
   >>> from toqito.channel_ops import partial_channel
   >>> from toqito.channels import depolarizing
   >>> rho = np.array([[0.3101, -0.0220-0.0219*1j, -0.0671-0.0030*1j, -0.0170-0.0694*1j],
   ...                 [-0.0220+0.0219*1j, 0.1008, -0.0775+0.0492*1j, -0.0613+0.0529*1j],
   ...                 [-0.0671+0.0030*1j, -0.0775-0.0492*1j, 0.1361, 0.0602 + 0.0062*1j],
   ...                 [-0.0170+0.0694*1j, -0.0613-0.0529*1j, 0.0602-0.0062*1j, 0.4530]])
   >>> partial_channel(rho, depolarizing(2), 1)
   array([[0.2231+0.j     , 0.0191-0.00785j, 0.    +0.j     ,
           0.    +0.j     ],
          [0.0191+0.00785j, 0.2769+0.j     , 0.    +0.j     ,
           0.    +0.j     ],
          [0.    +0.j     , 0.    +0.j     , 0.2231+0.j     ,
           0.0191-0.00785j],
          [0.    +0.j     , 0.    +0.j     , 0.0191+0.00785j,
           0.2769+0.j     ]])


   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If Phi map is not provided as a Choi matrix or Kraus
                       operators.
   :param rho: A matrix.
   :param phi_map: The map to partially apply.
   :param sys: Scalar or vector specifying the size of the subsystems.
   :param dim: Dimension of the subsystems. If :code:`None`, all dimensions
               are assumed to be equal.
   :return: The partial map :code:`phi_map` applied to matrix :code:`rho`.



