channel_ops.dual_channel
========================

.. py:module:: channel_ops.dual_channel

.. autoapi-nested-parse::

   Computes the dual of a map.



Functions
---------

.. autoapisummary::

   channel_ops.dual_channel.dual_channel


Module Contents
---------------

.. py:function:: dual_channel(phi_op, dims = None)

   Compute the dual of a map (quantum channel).

   (Section: Representations and Characterizations of Channels of :cite:`Watrous_2018_TQI`).

   The map can be represented as a Choi matrix, with optional specification of input
   and output dimensions. If the input channel maps :math:`M_{r,c}` to :math:`M_{x,y}`
   then :code:`dim` should be the list :code:`[[r,x], [c,y]]`. If it maps :math:`M_m`
   to :math:`M_n`, then :code:`dim` can simply be the vector :code:`[m,n]`. In this
   case the Choi matrix of the dual channel is returned, obtained by swapping input and
   output (see :func:`.swap`), and complex conjugating all elements.

   The map can also be represented as a list of Kraus operators.
   A list of lists, each containing two elements, corresponds to the families
   of operators :math:`\{(A_a, B_a)\}` representing the map

   .. math::
       \Phi(X) = \sum_a A_a X B^*_a.

   The dual map is obtained by taking the Hermitian adjoint of each operator.
   If :code:`phi_op` is given as a one-dimensional list, :math:`\{A_a\}`,
   it is interpreted as the completely positive map

   .. math::
       \Phi(X) = \sum_a A_a X A^*_a.

   .. rubric:: Examples

   When a channel is represented by a 1-D list of of Kraus operators, the CPTP dual channel can be determined
   as shown below.

   >>> import numpy as np
   >>> from toqito.channel_ops import dual_channel
   >>> kraus_1 = np.array([[1, 0, 1j, 0]])
   >>> kraus_2 = np.array([[0, 1, 0, 1j]])
   >>> kraus_list = [kraus_1, kraus_2]
   >>> dual_channel(kraus_list)
   [array([[1.-0.j],
          [0.-0.j],
          [0.-1.j],
          [0.-0.j]]), array([[0.-0.j],
          [1.-0.j],
          [0.-0.j],
          [0.-1.j]])]

   If the input channel's dimensions are different from the output dual channel's dimensions,

   >>> import numpy as np
   >>> from toqito.channel_ops import dual_channel
   >>> from toqito.perms import swap_operator
   >>> input_op = swap_operator([2, 3])
   >>> dual_channel(input_op, [[3, 2], [2, 3]])
   array([[1., 0., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0., 0.],
          [0., 0., 0., 0., 1., 0.],
          [0., 1., 0., 0., 0., 0.],
          [0., 0., 0., 1., 0., 0.],
          [0., 0., 0., 0., 0., 1.]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If matrices are not Choi matrix.
   :param phi_op: A superoperator. It should be provided either as a Choi matrix,
                  or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
   :param dims: Dimension of the input and output systems, for Choi matrix representation.
                If :code:`None`, try to infer them from :code:`phi_op.shape`.
   :return: The map dual to :code:`phi_op`, in the same representation.



