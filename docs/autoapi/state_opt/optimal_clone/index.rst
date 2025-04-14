state_opt.optimal_clone
=======================

.. py:module:: state_opt.optimal_clone

.. autoapi-nested-parse::

   Calculates success probability of approximately cloning a quantum state.



Functions
---------

.. autoapisummary::

   state_opt.optimal_clone.optimal_clone
   state_opt.optimal_clone.primal_problem
   state_opt.optimal_clone.dual_problem


Module Contents
---------------

.. py:function:: optimal_clone(states, probs, num_reps = 1, strategy = False)

   Compute probability of counterfeiting quantum money :cite:`Molina_2012_Optimal`.

   The primal problem for the :math:`n`-fold parallel repetition is given as follows:

   .. math::
       \begin{equation}
           \begin{aligned}
               \text{maximize:} \quad &
               \langle W_{\pi} \left(Q^{\otimes n} \right) W_{\pi}^*, X \rangle \\
               \text{subject to:} \quad & \text{Tr}_{\mathcal{Y}^{\otimes n}
                                          \otimes \mathcal{Z}^{\otimes n}}(X)
                                          = \mathbb{I}_{\mathcal{X}^{\otimes
                                          n}},\\
                                          & X \in \text{Pos}(
                                          \mathcal{Y}^{\otimes n}
                                          \otimes \mathcal{Z}^{\otimes n}
                                          \otimes \mathcal{X}^{\otimes n}).
           \end{aligned}
       \end{equation}

   The dual problem for the :math:`n`-fold parallel repetition is given as follows:

   .. math::
       \begin{equation}
           \begin{aligned}
               \text{minimize:} \quad & \text{Tr}(Y) \\
               \text{subject to:} \quad & \mathbb{I}_{\mathcal{Y}^{\otimes n}
               \otimes \mathcal{Z}^{\otimes n}} \otimes Y \geq W_{\pi}
               \left( Q^{\otimes n} \right) W_{\pi}^*, \\
               & Y \in \text{Herm} \left(\mathcal{X}^{\otimes n} \right).
           \end{aligned}
       \end{equation}

   .. rubric:: Examples

   Wiesner's original quantum money scheme :cite:`Wiesner_1983_Conjugate` was shown in :cite:`Molina_2012_Optimal`
   to have an optimal probability of 3/4 for succeeding a counterfeiting attack.

   Specifically, in the single-qubit case, Wiesner's quantum money scheme corresponds to the
   following ensemble:

   .. math::
       \left\{
           \left( \frac{1}{4}, |0\rangle \right),
           \left( \frac{1}{4}, |1\rangle \right),
           \left( \frac{1}{4}, |+\rangle \right),
           \left( \frac{1}{4}, |-\rangle \right)
       \right\},

   which yields the operator

   .. math::
       \begin{equation}
           Q = \frac{1}{4} \left(|000 \rangle \langle 000| + |111 \rangle \langle 111| +
                                 |+++ \rangle + \langle +++| + |--- \rangle \langle ---| \right).
       \end{equation}

   We can see that the optimal value we obtain in solving the SDP is 3/4.

   >>> from toqito.state_opt import optimal_clone
   >>> from toqito.states import basis
   >>> import numpy as np
   >>> e_0, e_1 = basis(2, 0), basis(2, 1)
   >>> e_p = (e_0 + e_1) / np.sqrt(2)
   >>> e_m = (e_0 - e_1) / np.sqrt(2)
   >>>
   >>> states = [e_0, e_1, e_p, e_m]
   >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
   >>> np.around(optimal_clone(states, probs), decimals=2)
   np.float64(0.75)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param states: A list of states provided as either matrices or vectors.
   :param probs: Respective list of probabilities each state is selected.
   :param num_reps: Number of parallel repetitions to perform.
   :param strategy: Boolean that denotes whether to return strategy.
   :return: The optimal probability with of counterfeiting quantum money.



.. py:function:: primal_problem(q_a, pperm, num_reps)

   Primal problem for counterfeit attack.

   As the primal problem takes longer to solve than the dual problem (as
   the variables are of larger dimension), the primal problem is only here
   for reference.

   :return: The optimal value of performing a counterfeit attack.


.. py:function:: dual_problem(q_a, pperm, num_reps)

   Dual problem for counterfeit attack.

   :return: The optimal value of performing a counterfeit attack.


