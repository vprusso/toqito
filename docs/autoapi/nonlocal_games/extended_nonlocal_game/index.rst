nonlocal_games.extended_nonlocal_game
=====================================

.. py:module:: nonlocal_games.extended_nonlocal_game

.. autoapi-nested-parse::

   Two-player extended nonlocal game.



Classes
-------

.. autoapisummary::

   nonlocal_games.extended_nonlocal_game.ExtendedNonlocalGame


Module Contents
---------------

.. py:class:: ExtendedNonlocalGame(prob_mat, pred_mat, reps = 1)

   Create two-player extended nonlocal game object.

   *Extended nonlocal games* are a superset of nonlocal games in which the
   players share a tripartite state with the referee. In such games, the
   winning conditions for Alice and Bob may depend on outcomes of measurements
   made by the referee, on its part of the shared quantum state, in addition
   to Alice and Bob's answers to the questions sent by the referee.

   Extended nonlocal games were initially defined in :cite:`Johnston_2016_Extended` and more
   information on these games can be found in :cite:`Russo_2017_Extended`.

   An example demonstration is available as a tutorial in the
   documentation. Go to :ref:`ref-label-bb84_extended_nl_example`.

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames


   .. py:method:: unentangled_value()

      Calculate the unentangled value of an extended nonlocal game.

      The *unentangled value* of an extended nonlocal game is the supremum
      value for Alice and Bob's winning probability in the game over all
      unentangled strategies. Due to convexity and compactness, it is possible
      to calculate the unentangled extended nonlocal game by:

      .. math::
          \omega(G) = \max_{f, g}
          \lVert
          \sum_{(x,y) \in \Sigma_A \times \Sigma_B} \pi(x,y)
          V(f(x), g(y)|x, y)
          \rVert

      where the maximum is over all functions :math:`f : \Sigma_A \rightarrow
      \Gamma_A` and :math:`g : \Sigma_B \rightarrow \Gamma_B`.

      :return: The unentangled value of the extended nonlocal game.



   .. py:method:: nonsignaling_value()

      Calculate the non-signaling value of an extended nonlocal game.

      The *non-signaling value* of an extended nonlocal game is the supremum
      value of the winning probability of the game taken over all
      non-signaling strategies for Alice and Bob.

      A *non-signaling strategy* for an extended nonlocal game consists of a
      function

      .. math::
          K : \Gamma_A \times \Gamma_B \times \Sigma_A \times \Sigma_B
          \rightarrow \text{Pos}(\mathcal{R})

      such that

      .. math::
          \sum_{a \in \Gamma_A} K(a,b|x,y) = \rho_b^y
          \quad \text{and} \quad
          \sum_{b \in \Gamma_B} K(a,b|x,y) = \sigma_a^x,

      for all :math:`x \in \Sigma_A` and :math:`y \in \Sigma_B` where
      :math:`\{\rho_b^y : y \in \Sigma_A, \ b \in \Gamma_B\}` and
      :math:`\{\sigma_a^x : x \in \Sigma_A, \ a \in \Gamma_B\}` are
      collections of operators satisfying

      .. math::
          \sum_{a \in \Gamma_A} \rho_b^y =
          \tau =
          \sum_{b \in \Gamma_B} \sigma_a^x,

      for every choice of :math:`x \in \Sigma_A` and :math:`y \in \Sigma_B`
      where :math:`\tau \in \text{D}(\mathcal{R})` is a density operator.

      :return: The non-signaling value of the extended nonlocal game.



   .. py:method:: quantum_value_lower_bound(iters = 5, tol = 1e-05)

      Calculate lower bound on the quantum value of an extended nonlocal game.

      Test

      :return: The quantum value of the extended nonlocal game.



   .. py:method:: __optimize_alice(bob_povms)

      Fix Bob's measurements and optimize over Alice's measurements.



   .. py:method:: __optimize_bob(rho)

      Fix Alice's measurements and optimize over Bob's measurements.



   .. py:method:: commuting_measurement_value_upper_bound(k = 1)

      Compute an upper bound on the commuting measurement value of an extended nonlocal game.

      This function calculates an upper bound on the commuting measurement value by
      using k-levels of the NPA hierarchy :cite:`Navascues_2008_AConvergent`. The NPA hierarchy is a uniform family
      of semidefinite programs that converges to the commuting measurement value of
      any extended nonlocal game.

      You can determine the level of the hierarchy by a positive integer or a string
      of a form like '1+ab+aab', which indicates that an intermediate level of the hierarchy
      should be used, where this example uses all products of one measurement, all products of
      one Alice and one Bob measurement, and all products of two Alice and one Bob measurements.

      .. rubric:: References

      .. bibliography::
          :filter: docname in docnames

      :param k: The level of the NPA hierarchy to use (default=1).
      :return: The upper bound on the commuting strategy value of an extended nonlocal game.



