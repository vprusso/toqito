state_opt.state_distinguishability
==================================

.. py:module:: state_opt.state_distinguishability

.. autoapi-nested-parse::

   Calculates the probability of optimally distinguishing quantum states.



Functions
---------

.. autoapisummary::

   state_opt.state_distinguishability.state_distinguishability
   state_opt.state_distinguishability._min_error_primal
   state_opt.state_distinguishability._min_error_dual
   state_opt.state_distinguishability._unambiguous_primal
   state_opt.state_distinguishability._unambiguous_dual


Module Contents
---------------

.. py:function:: state_distinguishability(vectors, probs = None, strategy = 'min_error', solver = 'cvxopt', primal_dual = 'dual', **kwargs)

   Compute probability of state distinguishability :cite:`Eldar_2003_SDPApproach`.

   The "quantum state distinguishability" problem involves a collection of :math:`n` quantum states

   .. math::
       \rho = \{ \rho_1, \ldots, \rho_n \},

   as well as a list of corresponding probabilities

   .. math::
       p = \{ p_1, \ldots, p_n \}.

   Alice chooses :math:`i` with probability :math:`p_i` and creates the state :math:`\rho_i`. Bob
   wants to guess which state he was given from the collection of states.

   For :code:`strategy = "min_error"`, this is the default method that yields the minimal
   probability of error for Bob.

   In that case, this function implements the following semidefinite program that provides the
   optimal probability with which Bob can conduct quantum state distinguishability.

   .. math::
       \begin{align*}
           \text{maximize:} \quad & \sum_{i=0}^n p_i \langle M_i, \rho_i \rangle \\
           \text{subject to:} \quad & M_0 + \ldots + M_n = \mathbb{I},\\
                                    & M_0, \ldots, M_n \geq 0.
       \end{align*}

   For :code:`strategy = "unambiguous"`, Bob never provides an incorrect answer, although it is
   possible that his answer is inconclusive.

   In that case, this function implements the following semidefinite program that provides the
   optimal probability with which Bob can conduct unambiguous quantum state distinguishability.

   .. math::
       \begin{align*}
           \text{maximize:} \quad & \mathbf{p} \cdot \mathbf{q} \\
           \text{subject to:} \quad & \Gamma - Q \geq 0,\\
                                    & \mathbf{q} \geq 0
       \end{align*}

   .. math::
       \begin{align*}
           \text{minimize:} \quad & \text{Tr}(\Gamma Z) \\
           \text{subject to:} \quad & z_i + p_i + \text{Tr}\left(F_iZ\right)=0,\\
                                    & Z, z \geq 0
       \end{align*}

   where :math:`\mathbf{p}` is the vector whose :math:`i`-th coordinate contains the probability
   that the state is prepared in state :math:`\left|\psi_i\right\rangle`, :math:`\Gamma` is
   the Gram matrix of :math:`\left|\psi_1\right\rangle,\cdots,\left|\psi_n\right\rangle` and :math:`F_i` is
   :math:`-|i\rangle\langle i|`.

   .. warning::
       Note that it only makes sense to distinguish unambiguously when the pure states are linearly
       independent. Calling this function on a set of states that doesn't verify this property will
       return 0.

   .. rubric:: Examples

   Minimal-error state distinguishability for the Bell states (which are perfectly distinguishable).

   >>> import numpy as np
   >>> from toqito.states import bell
   >>> from toqito.state_opt import state_distinguishability
   >>> states = [bell(0), bell(1), bell(2), bell(3)]
   >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
   >>> res, _ = state_distinguishability(vectors=states, probs=probs, primal_dual="dual")
   >>> np.around(res, decimals=2)
   np.float64(1.0)

   Note that if we are just interested in obtaining the optimal value, it is computationally less intensive to compute
   the dual problem over the primal problem. However, the primal problem does allow us to extract the explicit
   measurement operators which may be of interest to us.

   >>> import numpy as np
   >>> from toqito.states import bell
   >>> from toqito.state_opt import state_distinguishability
   >>> states = [bell(0), bell(1), bell(2), bell(3)]
   >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
   >>> res, measurements = state_distinguishability(vectors=states, probs=probs, primal_dual="primal")
   >>> np.around(measurements[0], decimals=5)  # doctest: +SKIP
   array([[ 0.5+0.j,  0. +0.j, -0. -0.j,  0.5-0.j],
          [ 0. -0.j,  0. +0.j, -0. +0.j,  0. -0.j],
          [-0. +0.j, -0. -0.j,  0. +0.j, -0. +0.j],
          [ 0.5+0.j,  0. +0.j, -0. -0.j,  0.5+0.j]])

   Unambiguous state distinguishability for unbiased states.

   >>> from toqito.state_opt import state_distinguishability
   >>> import numpy as np
   >>> states = [np.array([[1.], [0.]]), np.array([[1.],[1.]]) / np.sqrt(2)]
   >>> probs = [1 / 2, 1 / 2]
   >>> res, _ = state_distinguishability(vectors=states, probs=probs, primal_dual="primal", strategy="unambiguous")
   >>> np.around(res, decimals=2)
   np.float64(0.29)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param vectors: A list of states provided as vectors.
   :param probs: Respective list of probabilities each state is selected. If no
                 probabilities are provided, a uniform probability distribution is assumed.
   :param strategy: Whether to perform unambiguous or minimal error discrimination task. Possible
                    values are "min_error" and "unambiguous". Default option is `strategy="min_error"`.
   :param solver: Optimization option for `picos` solver. Default option is `solver="cvxopt"`.
   :param primal_dual: Option for the optimization problem. Default option is `"dual"`.
   :param kwargs: Additional arguments to pass to picos' solve method.
   :return: The optimal probability with which Bob can guess the state he was
            not given from `states` along with the optimal set of measurements.



.. py:function:: _min_error_primal(vectors, dim, probs = None, solver = 'cvxopt', **kwargs)

   Find the primal problem for minimum-error quantum state distinguishability SDP.


.. py:function:: _min_error_dual(vectors, dim, probs = None, solver = 'cvxopt', **kwargs)

   Find the dual problem for minimum-error quantum state distinguishability SDP.


.. py:function:: _unambiguous_primal(vectors, probs = None, solver = 'cvxopt', **kwargs)

   Solve the primal problem for unambiguous quantum state distinguishability SDP.

   Implemented according to Equation (5) of :cite:`Gupta_2024_Unambiguous`:.


.. py:function:: _unambiguous_dual(vectors, probs = None, solver = 'cvxopt', **kwargs)

   Solve the dual problem for unambiguous quantum state distinguishability SDP.

   Implemented according to Equation (5) of :cite:`Gupta_2024_Unambiguous`.


