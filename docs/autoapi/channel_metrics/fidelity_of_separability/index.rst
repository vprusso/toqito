channel_metrics.fidelity_of_separability
========================================

.. py:module:: channel_metrics.fidelity_of_separability

.. autoapi-nested-parse::

   Add functions for channel fidelity of Separability as defined in :cite:`Philip_2023_Schrodinger`.

   The constrainsts for this function are positive partial transpose (PPT)
   & k-extendible channels.



Functions
---------

.. autoapisummary::

   channel_metrics.fidelity_of_separability.fidelity_of_separability


Module Contents
---------------

.. py:function:: fidelity_of_separability(psi, psi_dims, k = 1, verbosity_option = 0, solver_option = 'cvxopt')

   Define the first benchmark introduced in Appendix I of :cite:`Philip_2023_Schrodinger`.

   If you would like to instead use the benchmark introduced in Appendix H,
   go to :obj:`toqito.state_metrics.fidelity_of_separability`.

   In :cite:`Philip_2023_Schrodinger` a variational quantum algorithm (VQA) is introduced to test
   the separability of a general bipartite state. The algorithm utilizes
   quantum steering between two separated systems such that the separability
   of the state is quantified.

   Due to the limitations of currently available quantum computers, two
   optimization semidefinite programs (SDP) benchmarks were introduced to
   maximize the fidelity of separability subject to some state constraints (Positive Partial Transpose (PPT),
   symmetric extensions (k-extendibility) :cite:`Hayden_2013_TwoMessage`).
   Entangled states do not have k-symmetric extensions. If an extension exists, it cannot be assumed directly
   that the state is separable. This function approximites the fidelity of separability by
   maximizing over PPT channels & k-extendible entanglement breaking channels
   i.e. an optimization problem over channels :cite:`Watrous_2018_TQI` .

   The following discussion (Equation (I4) from :cite:`Philip_2023_Schrodinger` ) defines the
   constraints for approximating :math:`\widetilde{F}_s^2(\rho_{AB})` in
   :math:`\frac{1}{2}(1+\widetilde{F}_s^2(\rho_{AB}))`.

   .. math::
       \operatorname{Tr}[
           \Pi_{A^{\prime}A}^{\operatorname{sym}} \operatorname{Tr}_{R}[
               T_R(\psi_{RAB})\Gamma^{\mathcal{E}^{k}}_{RA^{\prime}_1}]]

   Above expression defines the maximization problem subject to PPT & k-extendibile channel
   constraints over :math:`\max_{\Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}}\geq 0}`

   The constraint expressions are listed below:

   .. math::
       \operatorname{Tr}_{A^{\prime k}}[\Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}}]=I_R

   :math:`\Gamma^{\mathcal{E}^{k}}_{RA^{\prime}}` is Choi operator of
   entanglement breaking channel :math:`\mathcal{E}^{k}`.

   .. math::
       \Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}}= \mathcal{P}_{A^{\prime k}}(\Gamma^{\mathcal{E}^{k}}_{RA^{\prime k}})

   :math:`\mathcal{P}_{A^{\prime k}}` is the permutation operator over
   k-extensions :math:`A^{\prime k}`.

   .. math::
       T_{A^{\prime}_{1\cdots j}}(\Gamma^{\mathcal{E}^{k}_{RA^{\prime k}}}) \geq 0 \quad \forall j\leq k

   These other constraints are due to the PPT condition :cite:`Peres_1996_Separability`.

   .. rubric:: Examples

   Let's consider a density matrix of a state that we know is pure &
   separable. :math:`|000 \rangle = |0 \rangle \otimes |0 \rangle \otimes |0 \rangle`.

   The expected approximation of fidelity of separability is the maximum
   value possible i.e. very close to 1.

   .. math::
       \rho_{AB} = |000 \rangle \langle 000|

   >>> import numpy as np
   >>> from toqito.state_metrics import fidelity_of_separability
   >>> from toqito.matrix_ops import tensor
   >>> from toqito.states import basis
   >>> state = tensor(basis(2, 0), basis(2, 0))
   >>> rho = state @ state.conj().T
   >>> np.around(fidelity_of_separability(rho, [2, 2]), decimals=2)
   np.float64(1.0)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param psi: the density matrix for the tripartite state of interest psi_{BAR}
   :param psi_dims: the dimensions of System A, B, & R in
           the input state density matrix. It is assumed that the first
           quantity in this list is the dimension of System B.
   :param k: value for k-extendibility.
   :param verbosity_option: Parameter option for `picos`. Default value is
       `verbosity = 0`. For more info, visit
       https://picos-api.gitlab.io/picos/api/picos.modeling.options.html#option-verbosity.
   :param solver_option: Optimization option for `picos` solver. Default option is
       `solver_option="cvxopt"`. For more info, visit
       https://picos-api.gitlab.io/picos/api/picos.modeling.options.html#option-solver.
   :raises AssertionError: If the provided dimensions are not for a tripartite density matrix.
   :raises ValueError: If the matrix is not a density matrix (square matrix that
       is PSD with trace 1).
   :raises ValueError: the input state is entangled.
   :raises ValueError: the input state is a mixed state.
   :return: Optimized value of the SDP when maximized over a set of linear
       operators subject to some constraints.



