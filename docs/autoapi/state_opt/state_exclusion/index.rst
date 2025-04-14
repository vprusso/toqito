state_opt.state_exclusion
=========================

.. py:module:: state_opt.state_exclusion

.. autoapi-nested-parse::

   Calculates the probability of error of single state conclusive state exclusion.



Functions
---------

.. autoapisummary::

   state_opt.state_exclusion.state_exclusion
   state_opt.state_exclusion._min_error_primal
   state_opt.state_exclusion._min_error_dual
   state_opt.state_exclusion._unambiguous_primal
   state_opt.state_exclusion._unambiguous_dual


Module Contents
---------------

.. py:function:: state_exclusion(vectors, probs = None, strategy = 'min_error', solver = 'cvxopt', primal_dual = 'dual', **kwargs)

   Compute probability of error of single state conclusive state exclusion.

   The *quantum state exclusion* problem involves a collection of :math:`n` quantum states

   .. math::
       \rho = \{ \rho_0, \ldots, \rho_n \},

   as well as a list of corresponding probabilities

   .. math::
       p = \{ p_0, \ldots, p_n \}.

   Alice chooses :math:`i` with probability :math:`p_i` and creates the state :math:`\rho_i`.

   Bob wants to guess which state he was *not* given from the collection of states. State exclusion implies that
   ability to discard at least one out of the "n" possible quantum states by applying a measurement.

   For :code:`strategy = "min_error"`, this is the default method that yields the minimal probability of error for Bob.

   In that case, this function implements the following semidefinite program that provides the optimal probability
   with which Bob can conduct quantum state exclusion.

       .. math::
           \begin{equation}
               \begin{aligned}
                   \text{minimize:} \quad & \sum_{i=1}^n p_i \langle M_i, \rho_i \rangle \\
                   \text{subject to:} \quad & \sum_{i=1}^n M_i = \mathbb{I}_{\mathcal{X}}, \\
                                            & M_0, \ldots, M_n \in \text{Pos}(\mathcal{X}).
               \end{aligned}
           \end{equation}

       .. math::
           \begin{equation}
               \begin{aligned}
                   \text{maximize:} \quad & \text{Tr}(Y) \\
                   \text{subject to:} \quad & Y \preceq p_1\rho_1, \\
                                            & Y \preceq p_2\rho_2, \\
                                            & \vdots \\
                                            & Y \preceq p_n\rho_n, \\
                                            & Y \in\text{Herm}(\mathcal{X}).
               \end{aligned}
           \end{equation}

   For :code:`strategy = "unambiguous"`, Bob never provides an incorrect answer, although it is
   possible that his answer is inconclusive. This function then yields the probability of an inconclusive outcome.

   In that case, this function implements the following semidefinite program that provides the
   optimal probability with which Bob can conduct unambiguous quantum state distinguishability.

   .. math::
       \begin{align*}
           \text{minimize:} \quad & \text{Tr}\left(
               \left(\sum_{i=1}^n p_i\rho_i\right)\left(\mathbb{I}-\sum_{i=1}^nM_i\right)
               \right) \\
           \text{subject to:} \quad & \sum_{i=1}^nM_i \preceq \mathbb{I},\\
                                    & M_1, \ldots, M_n \succeq 0, \\
                                    & \langle M_1, \rho_1 \rangle, \ldots, \langle M_n, \rho_n \rangle =0
       \end{align*}

   .. math::
       \begin{align*}
           \text{maximize:} \quad & 1 - \text{Tr}(N) \\
           \text{subject to:} \quad & a_1p_1\rho_1, \ldots, a_np_n\rho_n \succeq \sum_{i=1}^np_i\rho_i - N,\\
                                    & N \succeq 0,\\
                                    & a_1, \ldots, a_n \in\mathbb{R}
       \end{align*}


   .. note::
       It is known that it is always possible to perfectly exclude pure states that are linearly dependent.
       Thus, calling this function on a set of states with this property will return 0.

   The conclusive state exclusion SDP is written explicitly in :cite:`Bandyopadhyay_2014_Conclusive`. The problem
   of conclusive state exclusion was also thought about under a different guise in :cite:`Pusey_2012_On`.

   .. rubric:: Examples

   Consider the following two Bell states

   .. math::
       \begin{equation}
           \begin{aligned}
               u_0 &= \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right), \\
               u_1 &= \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right).
           \end{aligned}
       \end{equation}

   It is not possible to conclusively exclude either of the two states. We can see that the result of the function in
   :code:`|toqito⟩` yields a value of :math:`0` as the probability for this to occur.

   >>> from toqito.state_opt import state_exclusion
   >>> from toqito.states import bell
   >>> import numpy as np
   >>>
   >>> vectors = [bell(0), bell(1)]
   >>> probs = [1/2, 1/2]
   >>>
   >>> np.around(state_exclusion(vectors, probs)[0], decimals=2)
   np.float64(0.0)

   Unambiguous state exclusion for unbiased states.

   >>> from toqito.state_opt import state_exclusion
   >>> import numpy as np
   >>> states = [np.array([[1.], [0.]]), np.array([[1.],[1.]]) / np.sqrt(2)]
   >>> res, _ = state_exclusion(states, primal_dual="primal", strategy="unambiguous", abs_ipm_opt_tol=1e-7)
   >>> np.around(res, decimals=2)
   np.float64(0.71)

   .. note::
       If you encounter a `ZeroDivisionError` or an `ArithmeticError` when using cvxopt as a solver (which is the
       default), you might want to set the `abs_ipm_opt_tol` option to a lower value (the default being `1e-8`) or
       to set the `cvxopt_kktsolver` option to `ldl`.

       See https://gitlab.com/picos-api/picos/-/issues/341

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param vectors: A list of states provided as vectors.
   :param probs: Respective list of probabilities each state is selected. If no
                 probabilities are provided, a uniform probability distribution is assumed.
   :param strategy: Whether to perform minimal error or unambiguous discrimination task. Possible values are
                    "min_error" and "unambiguous".
   :param solver: Optimization option for `picos` solver. Default option is `solver_option="cvxopt"`.
   :param primal_dual: Option for the optimization problem.
   :param kwargs: Additional arguments to pass to picos' solve method.
   :return: The optimal probability with which Bob can guess the state he was
            not given from `states` along with the optimal set of measurements.



.. py:function:: _min_error_primal(vectors, dim, probs = None, solver = 'cvxopt', **kwargs)

   Find the primal problem for minimum-error quantum state exclusion SDP.


.. py:function:: _min_error_dual(vectors, dim, probs = None, solver = 'cvxopt', **kwargs)

   Find the dual problem for minimum-error quantum state exclusion SDP.


.. py:function:: _unambiguous_primal(vectors, dim, probs = None, solver = 'cvxopt', **kwargs)

   Solve the primal problem for unambiguous quantum state distinguishability SDP.

   Implemented according to Equation (33) of :cite:`Bandyopadhyay_2014_Conclusive`.


.. py:function:: _unambiguous_dual(vectors, dim, probs = None, solver = 'cvxopt', **kwargs)

   Solve the dual problem for unambiguous quantum state distinguishability SDP.

   Implemented according to Equation (35) of :cite:`Bandyopadhyay_2014_Conclusive`.


