channel_ops.natural_representation
==================================

.. py:module:: channel_ops.natural_representation

.. autoapi-nested-parse::

   Kraus operators to natural representation.



Functions
---------

.. autoapisummary::

   channel_ops.natural_representation.natural_representation


Module Contents
---------------

.. py:function:: natural_representation(kraus_ops)

   Convert a set of Kraus operators to the natural representation of a quantum channel.

   The natural representation of a quantum channel is given by:
   :math:`\Phi = \sum_i K_i \otimes K_i^*`
   where :math:`K_i^*` is the complex conjugate of :math:`K_i`.

   .. rubric:: Examples

   >>> import numpy as np
   >>> from toqito.channel_ops import natural_representation
   >>> k0 = np.sqrt(1/2) * np.array([[1, 0], [0, 1]])
   >>> k1 = np.sqrt(1/2) * np.array([[0, 1], [1, 0]])
   >>> print(natural_representation([k0, k1]))
   [[0.5 0.  0.  0.5]
    [0.  0.5 0.5 0. ]
    [0.  0.5 0.5 0. ]
    [0.5 0.  0.  0.5]]


