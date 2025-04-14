nonlocal_games.nonlocal_game
============================

.. py:module:: nonlocal_games.nonlocal_game

.. autoapi-nested-parse::

   Two-player nonlocal game.



Classes
-------

.. autoapisummary::

   nonlocal_games.nonlocal_game.NonlocalGame


Module Contents
---------------

.. py:class:: NonlocalGame(prob_mat, pred_mat, reps = 1)

   Create two-player nonlocal game object.

   *Nonlocal games* are a mathematical framework that abstractly models a
   physical system. This game is played between two players, Alice and Bob, who
   are not allowed to communicate with each other once the game has started and
   who play cooperative against an adversary referred to as the referee.

   The nonlocal game framework was originally introduced in :cite:`Cleve_2010_Consequences`.

   A tutorial is available in the documentation. For more info, see :ref:`ref-label-nl-games-tutorial`.

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames


   .. py:method:: from_bcs_game(constraints, reps = 1)
      :classmethod:


      Convert constraints that specify a binary constraint system game to a nonlocal game.

      Binary constraint system games (BCS) games were originally defined in :cite:`Cleve_2014_Characterization`.

      :param constraints: List of binary constraints that define the game.
      :param reps: Number of parallel repetitions to perform. Default is 1.
      :return: A NonlocalGame object arising from the variables and constraints that define the game.



   .. py:method:: process_iteration(num_bob_outputs, num_bob_inputs, pred_mat_copy, num_alice_outputs, num_alice_inputs)

      Help the classical_value function as a helper method.

      :return: A value between [0, 1] representing the tgval.



   .. py:method:: classical_value()

      Compute the classical value of the nonlocal game.

      This function has been adapted from the QETLAB package.

      :return: A value between [0, 1] representing the classical value.



   .. py:method:: quantum_value_lower_bound(dim = 2, iters = 5, tol = 1e-05)

      Compute a lower bound on the quantum value of a nonlocal game :cite:`Liang_2007_Bounds`.

      Calculates a lower bound on the maximum value that the specified
      nonlocal game can take on in quantum mechanical settings where Alice and
      Bob each have access to `dim`-dimensional quantum system.

      This function works by starting with a randomly-generated POVM for Bob,
      and then optimizing Alice's POVM and the shared entangled state. Then
      Alice's POVM and the entangled state are fixed, and Bob's POVM is
      optimized. And so on, back and forth between Alice and Bob until
      convergence is reached.

      Note that the algorithm is not guaranteed to obtain the optimal local
      bound and can get stuck in local minimum values. The alleviate this, the
      `iter` parameter allows one to run the algorithm some pre-specified
      number of times and keep the highest value obtained.

      The algorithm is based on the alternating projections algorithm as it
      can be applied to Bell inequalities as shown in :cite:`Liang_2007_Bounds`.

      The alternating projection algorithm has also been referred to as the
      "see-saw" algorithm as it goes back and forth between the following two
      semidefinite programs:

      .. math::

          \begin{equation}
              \begin{aligned}
                  \textbf{SDP-1:} \quad & \\
                  \text{maximize:} \quad & \sum_{(x,y \in \Sigma)} \pi(x,y)
                                           \sum_{(a,b) \in \Gamma}
                                           V(a,b|x,y)
                                           \langle B_b^y, A_a^x \rangle \\
                  \text{subject to:} \quad & \sum_{a \in \Gamma_{\mathsf{A}}}=
                                      \tau, \qquad \qquad
                                      \forall x \in \Sigma_{\mathsf{A}}, \\
                                     \quad & A_a^x \in \text{Pos}(\mathcal{A}),
                                      \qquad
                                      \forall x \in \Sigma_{\mathsf{A}}, \
                                      \forall a \in \Gamma_{\mathsf{A}}, \\
                                      & \tau \in \text{D}(\mathcal{A}).
              \end{aligned}
          \end{equation}

      .. math::

          \begin{equation}
              \begin{aligned}
                  \textbf{SDP-2:} \quad & \\
                  \text{maximize:} \quad & \sum_{(x,y \in \Sigma)} \pi(x,y)
                                           \sum_{(a,b) \in \Gamma} V(a,b|x,y)
                                           \langle B_b^y, A_a^x \rangle \\
                  \text{subject to:} \quad & \sum_{b \in \Gamma_{\mathsf{B}}}=
                                      \mathbb{I}, \qquad \qquad
                                      \forall y \in \Sigma_{\mathsf{B}}, \\
                                  \quad & B_b^y \in \text{Pos}(\mathcal{B}),
                                  \qquad \forall y \in \Sigma_{\mathsf{B}}, \
                                  \forall b \in \Gamma_{\mathsf{B}}.
              \end{aligned}
          \end{equation}

      .. rubric:: Examples

      The CHSH game

      The CHSH game is a two-player nonlocal game with the following
      probability distribution and question and answer sets.

      .. math::
          \begin{equation}
          \begin{aligned}
            \pi(x,y) = \frac{1}{4}, \qquad (x,y) \in \Sigma_A \times \Sigma_B,
            \qquad \text{and} \qquad (a, b) \in \Gamma_A \times \Gamma_B,
          \end{aligned}
          \end{equation}

      where

      .. math::
          \begin{equation}
          \Sigma_A = \{0, 1\}, \quad \Sigma_B = \{0, 1\}, \quad \Gamma_A =
          \{0,1\}, \quad \text{and} \quad \Gamma_B = \{0, 1\}.
          \end{equation}

      Alice and Bob win the CHSH game if and only if the following equation is
      satisfied.

      .. math::
          \begin{equation}
          a \oplus b = x \land y.
          \end{equation}

      Recall that :math:`\oplus` refers to the XOR operation.

      The optimal quantum value of CHSH is
      :math:`\cos(\pi/8)^2 \approx 0.8536` where the optimal classical value
      is :math:`3/4`.

      >>> import numpy as np
      >>> from toqito.nonlocal_games.nonlocal_game import NonlocalGame
      >>>
      >>> dim = 2
      >>> num_alice_inputs, num_alice_outputs = 2, 2
      >>> num_bob_inputs, num_bob_outputs = 2, 2
      >>> prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
      >>> pred_mat = np.zeros((num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs))
      >>>
      >>> for a_alice in range(num_alice_outputs):
      ...     for b_bob in range(num_bob_outputs):
      ...        for x_alice in range(num_alice_inputs):
      ...            for y_bob in range(num_bob_inputs):
      ...                if np.mod(a_alice + b_bob + x_alice * y_bob, dim) == 0:
      ...                    pred_mat[a_alice, b_bob, x_alice, y_bob] = 1
      >>>
      >>> chsh = NonlocalGame(prob_mat, pred_mat)
      >>> chsh.quantum_value_lower_bound()   # doctest: +SKIP
      0.85

      .. rubric:: References

      .. bibliography::
          :filter: docname in docnames

      :param dim: The dimension of the quantum system that Alice and Bob have
                  access to (default = 2).
      :param iters: The number of times to run the alternating projection
                    algorithm.
      :param tol: The tolerance before quitting out of the alternating
                  projection semidefinite program.
      :return: The lower bound on the quantum value of a nonlocal game.




   .. py:method:: __optimize_alice(dim, bob_povms)

      Fix Bob's measurements and optimize over Alice's measurements.



   .. py:method:: __optimize_bob(dim, alice_povms)

      Fix Alice's measurements and optimize over Bob's measurements.



   .. py:method:: nonsignaling_value()

      Compute the non-signaling value of the nonlocal game.

      :return: A value between [0, 1] representing the non-signaling value.



   .. py:method:: commuting_measurement_value_upper_bound(k = 1)

      Compute an upper bound on the commuting measurement value of the nonlocal game.

      This function calculates an upper bound on the commuting measurement value by
      using k-levels of the NPA hierarchy :cite:`Navascues_2008_AConvergent`. The NPA hierarchy is a uniform family
      of semidefinite programs that converges to the commuting measurement value of
      any nonlocal game.

      You can determine the level of the hierarchy by a positive integer or a string
      of a form like '1+ab+aab', which indicates that an intermediate level of the hierarchy
      should be used, where this example uses all products of one measurement, all products of
      one Alice and one Bob measurement, and all products of two Alice and one Bob measurements.

      .. rubric:: References

      .. bibliography::
          :filter: docname in docnames

      :param k: The level of the NPA hierarchy to use (default=1).
      :return: The upper bound on the commuting strategy value of a nonlocal game.



