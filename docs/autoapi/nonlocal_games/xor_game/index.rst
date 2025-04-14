nonlocal_games.xor_game
=======================

.. py:module:: nonlocal_games.xor_game

.. autoapi-nested-parse::

   Two-player XOR game.



Classes
-------

.. autoapisummary::

   nonlocal_games.xor_game.XORGame


Module Contents
---------------

.. py:class:: XORGame(prob_mat, pred_mat, reps = 1, tol = None)

   Create two-player XOR game object.

   Calculates the optimal probability that Alice and Bob win the game if they
   are allowed to determine a join strategy beforehand, but not allowed to
   communicate during the game itself.

   The quantum value of an XOR game can be solved via the semidefinite program
   from :cite:`Cleve_2010_Consequences`.

   This function is adapted from the QETLAB package.

   A tutorial is available in the documentation. Go to :ref:`ref-label-xor-quantum-value-tutorial`.

   .. rubric:: Examples

   The CHSH game

   The CHSH game is a two-player nonlocal game with the following probability
   distribution and question and answer sets :cite:`Cleve_2008_Strong`.

   .. math::
           \begin{equation}
                   \begin{aligned} \pi(x,y) = \frac{1}{4}, \qquad (x,y) \in
                                                   \Sigma_A \times
                           \Sigma_B, \qquad \text{and} \qquad (a, b) \in \Gamma_A \times
                           \Gamma_B,
                   \end{aligned}
           \end{equation}

   where

   .. math::
           \begin{equation}
                   \Sigma_A = \{0, 1\}, \quad \Sigma_B = \{0, 1\}, \quad \Gamma_A =
                   \{0,1\}, \quad \text{and} \quad \Gamma_B = \{0, 1\}.
           \end{equation}

   Alice and Bob win the CHSH game if and only if the following equation is
   satisfied

   .. math::
           \begin{equation}
           a \oplus b = x \land y.
           \end{equation}

   Recall that :math:`\oplus` refers to the XOR operation.

   The optimal quantum value of CHSH is :math:`\cos(\pi/8)^2 \approx 0.8536`
   where the optimal classical value is :math:`3/4`.

   In order to specify the CHSH game, we can define the probability matrix and
   predicate matrix for the CHSH game as `numpy` arrays as follows.

   >>> import numpy as np
   >>> prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
   >>> pred_mat = np.array([[0, 0], [0, 1]])

   In `toqito`, we can calculate both the quantum and classical value of the
   CHSH game as follows.

   >>> import numpy as np
   >>> from toqito.nonlocal_games.xor_game import XORGame
   >>> chsh = XORGame(prob_mat, pred_mat)
   >>> np.around(chsh.quantum_value(), decimals=2)
   np.float64(0.85)
   >>>
   >>> chsh.classical_value()
   np.float64(0.75)

   The odd cycle game

   The odd cycle game is another XOR game :cite:`Cleve_2010_Consequences`. For this game, we can
   specify the probability and predicate matrices as follows.

   >>> prob_mat = np.array(
   ... [
   ...     [0.1, 0.1, 0, 0, 0],
   ...     [0, 0.1, 0.1, 0, 0],
   ...     [0, 0, 0.1, 0.1, 0],
   ...     [0, 0, 0, 0.1, 0.1],
   ...     [0.1, 0, 0, 0, 0.1],
   ... ]
   ... )
   >>> pred_mat = np.array(
   ... [
   ...     [0, 1, 0, 0, 0],
   ...     [0, 0, 1, 0, 0],
   ...     [0, 0, 0, 1, 0],
   ...     [0, 0, 0, 0, 1],
   ...     [1, 0, 0, 0, 0],
   ... ]
   ... )

   In :code:`|toqito⟩`, we can calculate both the quantum and classical value of
   the odd cycle game as follows.

   >>> import numpy as np
   >>> from toqito.nonlocal_games.xor_game import XORGame
   >>> odd_cycle = XORGame(prob_mat, pred_mat)
   >>> np.around(odd_cycle.quantum_value(), decimals=2)
   np.float64(0.98)
   >>> np.around(odd_cycle.classical_value(), decimals=1)
   np.float64(0.9)

   We can also calculate the nonsignaling value of the odd cycle game.
   >>> np.around(odd_cycle.nonsignaling_value(), decimals=1)
   np.float64(1.0)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames


   .. py:attribute:: prob_mat


   .. py:attribute:: pred_mat


   .. py:attribute:: reps
      :value: 1



   .. py:method:: quantum_value()

      Compute the quantum value of the XOR game.

      To obtain the quantum value of the XOR game, we calculate the following
      simplified dual problem of the semidefinite program from the set of
      notes: Lecture 6 of :cite:`Watrous_2011_Lecture_Notes`

              .. math::
                      \begin{equation}
                              \begin{aligned}
                                      \text{minimize:} \quad & \frac{1}{2} \sum_{x \in X} u(x) +
                                                                                       \frac{1}{2} \sum_{
                                                                                          y \in Y} v(y) \\
                                      \text{subject to:} \quad &
                                                      \begin{pmatrix}
                                                              \text{Diag}(u) & -D \\
                                                              -D^* & \text{Diag}(v)
                                                      \end{pmatrix} \geq 0, \\
                                                      & u \in \mathbb{R}^X, \
                                                        v \in \mathbb{R}^Y.
                              \end{aligned}
                      \end{equation}

              where :math:`D` is the matrix defined to be

              .. math::
                      D(x,y) = \pi(x, y) (-1)^{f(x,y)}

              In other words, :math:`\pi(x, y)` corresponds to :code:`prob_mat[x, y]`,
              and :math:`f(x,y)` corresponds to :code:`pred_mat[x, y]`.

              :return: A value between [0, 1] representing the quantum value.



   .. py:method:: classical_value()

      Compute the classical value of the XOR game.

      :return: A value between [0, 1] representing the classical value.



   .. py:method:: nonsignaling_value()

      Compute the nonsignaling value of an XOR game.

      Here, the exising function in the :code:`NonlocalGame` class is called.

      :return: A value between [0, 1] representing the nonsignaling value.



   .. py:method:: to_nonlocal_game()

      Given an XOR game, compute a predicate matrix representing the more generic :code:`NonlocalGame` equivalent.

      :return: A :code:`NonlocalGame` object equivalent to the XOR game.



