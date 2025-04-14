state_props.schmidt_rank
========================

.. py:module:: state_props.schmidt_rank

.. autoapi-nested-parse::

   Calculate the Schmidt rank of a quantum state.



Functions
---------

.. autoapisummary::

   state_props.schmidt_rank.schmidt_rank
   state_props.schmidt_rank._operator_schmidt_rank


Module Contents
---------------

.. py:function:: schmidt_rank(rho, dim = None)

   Compute the Schmidt rank :cite:`WikiScmidtDecomp`.

   For complex Euclidean spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}`, a pure state
   :math:`u \in \mathcal{X} \otimes \mathcal{Y}` possesses an expansion of the form:

   .. math::
       u = \sum_{i} \lambda_i v_i w_i,

   where :math:`v_i \in \mathcal{X}` and :math:`w_i \in \mathcal{Y}` are orthonormal states.

   The Schmidt coefficients are calculated from

   .. math::
       A = \text{Tr}_{\mathcal{B}}(u^* u).

   The Schmidt rank is the number of non-zero eigenvalues of :math:`A`. The Schmidt rank allows us
   to determine if a given state is entangled or separable. For instance:

       - If the Schmidt rank is 1: The state is separable,
       - If the Schmidt rank > 1: The state is entangled.

   Compute the Schmidt rank of the input :code:`rho`, provided as either a vector or a matrix that
   is assumed to live in bipartite space, where both subsystems have dimension equal to
   :code:`sqrt(len(vec))`.

   The dimension may be specified by the 1-by-2 vector :code:`dim` and the rank in that case is
   determined as the number of Schmidt coefficients larger than :code:`tol`.

   .. rubric:: Examples

   Computing the Schmidt rank of the entangled Bell state should yield a value greater than one.

   >>> from toqito.states import bell
   >>> from toqito.state_props import schmidt_rank
   >>> rho = bell(0) @ bell(0).conj().T
   >>> schmidt_rank(rho)
   np.int64(4)

   Computing the Schmidt rank of the entangled singlet state should yield a value greater than
   :math:`1`.

   >>> from toqito.states import bell
   >>> from toqito.state_props import schmidt_rank
   >>> u = bell(2) @ bell(2).conj().T
   >>> schmidt_rank(u)
   np.int64(4)

   Computing the Schmidt rank of a separable state should yield a value equal to :math:`1`.

   >>> from toqito.states import basis
   >>> from toqito.state_props import schmidt_rank
   >>> import numpy as np
   >>> e_0, e_1 = basis(2, 0), basis(2, 1)
   >>> e_00 = np.kron(e_0, e_0)
   >>> e_01 = np.kron(e_0, e_1)
   >>> e_10 = np.kron(e_1, e_0)
   >>> e_11 = np.kron(e_1, e_1)
   >>>
   >>> rho = 1 / 2 * (e_00 - e_01 - e_10 + e_11)
   >>> rho = rho @ rho.conj().T
   >>> schmidt_rank(rho)
   np.int64(1)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param rho: A bipartite vector or matrix to have its Schmidt rank computed.
   :param dim: A 1-by-2 vector or matrix.
   :return: The Schmidt rank of :code:`rho`.



.. py:function:: _operator_schmidt_rank(rho, dim = None)

   Operator Schmidt rank of variable.

   If the input is provided as a density operator instead of a vector, compute
   the operator Schmidt rank.


