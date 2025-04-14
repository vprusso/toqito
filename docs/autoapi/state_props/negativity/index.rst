state_props.negativity
======================

.. py:module:: state_props.negativity

.. autoapi-nested-parse::

   Calculates the negativity property of a quantum state.



Functions
---------

.. autoapisummary::

   state_props.negativity.negativity


Module Contents
---------------

.. py:function:: negativity(rho, dim = None)

   Compute the negativity of a bipartite quantum state :cite:`WikiNeg`.

   The negativity of a subsystem can be defined in terms of a density matrix :math:`\rho`:

   .. math::
       \mathcal{N}(\rho) \equiv \frac{||\rho^{\Gamma_A}||_1-1}{2}.

   Calculate the negativity of the quantum state :math:`\rho`, assuming that the two subsystems on
   which :math:`\rho` acts are of equal dimension (if the local dimensions are unequal, specify
   them in the optional :code:`dim` argument). The negativity of :math:`\rho` is the sum of the
   absolute value of the negative eigenvalues of the partial transpose of :math:`\rho`.

   .. rubric:: Examples

   Example of the negativity of density matrix of Bell state.

   >>> from toqito.states import bell
   >>> from toqito.state_props import negativity
   >>> rho = bell(0) @ bell(0).conj().T
   >>> negativity(rho)
   np.float64(0.4999999999999998)

   .. seealso:: :func:`.log_negativity`

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If dimension of matrix is invalid.
   :param rho: A density matrix of a pure state vector.
   :param dim: The default has both subsystems of equal dimension.
   :return: A value between 0 and 1 that corresponds to the negativity of :math:`\rho`.



