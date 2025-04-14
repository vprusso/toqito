state_props.log_negativity
==========================

.. py:module:: state_props.log_negativity

.. autoapi-nested-parse::

   Calculates the logarithmic negativity property of a quantum state.



Functions
---------

.. autoapisummary::

   state_props.log_negativity.log_negativity


Module Contents
---------------

.. py:function:: log_negativity(rho, dim = None)

   Compute the log-negativity of a bipartite quantum state :cite:`WikiNeg`.

   The log-negativity of a subsystem can be defined in terms of a density matrix :math:`\rho`:

   .. math::
       E_\mathcal{N}(\rho) \equiv \text{log}_2\left( ||\rho^{\Gamma_A}||_1 \right).

   Calculate the log-negativity of the quantum state :math:`\rho`, assuming that the two subsystems
   on which :math:`\rho` acts are of equal dimension (if the local dimensions are unequal, specify
   them in the optional :code:`dim` argument).

   .. rubric:: Examples

   Example of the log-negativity of density matrix of Bell state.

   >>> from toqito.states import bell
   >>> from toqito.state_props import log_negativity
   >>> rho = bell(0) @ bell(0).conj().T
   >>> log_negativity(rho)
   np.float64(0.9999999999999997)

   .. seealso:: :func:`.negativity`

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If the input matrix is not a density matrix.
   :param rho: A density matrix of a pure state vector.
   :param dim: The default has both subsystems of equal dimension.
   :return: A positive value that corresponds to the logarithmic negativity of :math:`\rho`.



