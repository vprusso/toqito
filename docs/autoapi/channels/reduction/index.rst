channels.reduction
==================

.. py:module:: channels.reduction

.. autoapi-nested-parse::

   Generates the reduction channel.



Functions
---------

.. autoapisummary::

   channels.reduction.reduction


Module Contents
---------------

.. py:function:: reduction(dim, k = 1)

   Produce the reduction map or reduction channel :cite:`WikiReductionCrit`.

   If :code:`k = 1`, this returns the Choi matrix of the reduction map which is a positive map
   on :code:`dim`-by-:code:`dim` matrices. For a different value of :code:`k`, this yields the
   Choi matrix of the map defined by:

   .. math::
       R(X) = k * \text{Tr}(X) * \mathbb{I} - X,

   where :math:`\mathbb{I}` is the identity matrix. This map is :math:`k`-positive.

   .. rubric:: Examples

   Using :code:`|toqito⟩`, we can generate the :math:`3`-dimensional (or standard) reduction map
   as follows.

   >>> from toqito.channels import reduction
   >>> reduction(3)
   array([[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.],
          [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
          [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
          [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
          [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
          [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
          [-1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param dim: A positive integer (the dimension of the reduction map).
   :param k: If this positive integer is provided, the script will instead return the Choi
             matrix of the following linear map: Phi(X) := K * Tr(X)I - X.
   :return: The reduction map.


