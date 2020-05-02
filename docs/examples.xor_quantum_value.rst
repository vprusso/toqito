Calculating the Quantum and Classical Value of a Two-Player XOR Game
=====================================================================

In this tutorial, we will cover the concept of an *XOR game*. We will also
showcase how the `toqito` software package can be used to calculate the
classical and quantum value of a given XOR game.

For readers who are already familiar with XOR games and who simply want to see
how to use `toqito` to study these objects, they are welcome to consult the
documentation page, and more specifically the function `xor\_game\_value
<https://toqito.readthedocs.io/en/latest/nonlocal_games.xor_games.html>`_.

Two-player XOR games
--------------------

A two-player XOR game is a nonlocal game in which the winning condition is
predicated on an XOR function. For more information on the more general class
of nonlocal games along with how one defines classical and quantum strategies
for these games, please refer to the example:

* `Lower Bounds on the Quantum Value of a Two-Player Nonlocal Game
  <https://toqito.readthedocs.io/en/latest/examples.nonlocal_quantum_lower_bound.html>`_

.. note::
    It is *not* known how to directly compute the quantum value of an arbitrary
    nonlocal game. For the subset of XOR games, it turns out that it is
    possible to directly calculate the quantum value by solving a semidefinite
    program. The `toqito` package obtains the quantum value of an XOR game in
    this manner.

The rest of this tutorial is concerned with analyzying specific XOR games.

The CHSH game
-------------

The *CHSH game* is a two-player XOR game with the following probability
distribution and question and answer sets.

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

For each question secnario, the following table provides what the winning
condition must be equal to for each question tuple to induce a winning outcome.

.. table::
    :align: center

    +-------------+-------------+----------------------+
    | :math:`x`   | :math:`y`   |  :math:`a \oplus b`  |
    +=============+=============+======================+
    | :math:`0`   | :math:`0`   | :math:`0`            |
    +-------------+-------------+----------------------+
    | :math:`0`   | :math:`1`   | :math:`1`            |
    +-------------+-------------+----------------------+
    | :math:`1`   | :math:`0`   | :math:`1`            |
    +-------------+-------------+----------------------+
    | :math:`1`   | :math:`1`   | :math:`0`            |
    +-------------+-------------+----------------------+

In order to specify an XOR game in `toqito`, we will define two matrices:

    * `prob_mat`: A matrix whose :math:`(x, y)^{th}` entry corresponds to
      the probablity that Alice receives question :math:`x` and Bob receives
      question :math:`y`.

    * `pred_mat`: A matrix whose :math:`(x, y)^{th}` entry corresponds
      to the winning choice of :math:`a` and :math:`b` when Alice receives
      :math:`x` and Bob receives :math:`y` from the referee.

For the CHSH game, the `prob_mat` and `pred_mat` variables are defined as follows.

.. code-block:: python

    import numpy as np
    prob_mat = np.array([[1/4, 1/4], 
                         [1/4, 1/4]])
    pred_mat = np.array([[0, 0],
                         [0, 1]])

That is, the `prob_mat` matrix encapsulates that each question pair
:math:`\{(0,0), (0, 1), (1, 0), (1, 1)\}` is equally likely. 

The `pred_mat` matrix indicates what the winning outcome of Alice and Bob
should be. For instance, `pred_mat[0][0] = 0` describes the scenario where
Alice and Bob both recieve :math:`0` as input. As we want to satisfy the
winning condition :math:`x \land y = a \oplus b`, we must have that :math:`a
\oplus b = 0` to satisfy the case when both :math:`x` and :math:`y` are equal
to zero. A similar logic can be followed to populate the remaining entries of
the `pred_mat` variable.

We will use both of the `prob_mat` and `pred_mat` variables in the coming
subsections to make use of the `toqito` package to compute both the classical
and quantum value of the CHSH game.

A classical strategy for the CHSH game
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can begin by asking; is it possible for Alice and Bob to win for every
single question pair they receive with certainty? If Alice and Bob use a
classical strategy, the answer to this question is "no". To see why, consider
the following equations:

.. math::
    \begin{equation}
        \begin{aligned}
            a_0 \oplus b_0 = 0, \quad a_0 \oplus b_1 = 0, \\
            a_1 \oplus b_0 = 0, \quad a_1 \oplus b_1 = 1.
        \end{aligned}
    \end{equation}

In the above equation, :math:`a_x` is Alice's answer in the event that she
receives question :math:`x` from the referee for :math:`x \in \Sigma_A`.
Similarly, :math:`b_y` is Bob's answer when Bob receives question :math:`y`
from the referee for :math:`y \in \Sigma_B`. These equations express the
winning conditions that Alice and Bob must satisfy in order to perfectly win
the CHSH game. That is, if it's possible to satisfy all of these equations
simultaneously, it's not possible for them to lose. 

One could perform a brute-force check to see that there is no possible way for
Alice and Bob to simultaneously satisfy all four equations. The best they can
do is satisfy three out of the four equations 

.. math::
    \begin{equation}
        \begin{aligned}
            a_0 \oplus b_0 = 0, \quad a_0 \oplus b_1 = 0, \\
            a_1 \oplus b_0 = 0.
        \end{aligned}
    \end{equation}

They can achieve this if they either have answers :math:`a_0 = b_0 = a_1 = b_1
= 0` or :math:`a_0 = b_0 = a_1 = b_1 = 1`.

Since it is not possible to satisfy all four equations, but it is possible to
satisfy three out of the four equations, the classical value of the CHSH game
is :math:`3/4`, or stated in an equivalent way

.. math::
    \begin{equation}
        \omega(G_{CHSH}) = 3/4 = 0.75.
    \end{equation}

We can verify this by making use of `toqito` to compute the classical value of the CHSH game.


.. code-block:: python

    import numpy as np
    import toqito as tq
    prob_mat = np.array([[1/4, 1/4], 
                         [1/4, 1/4]])
    pred_mat = np.array([[0, 0],
                         [0, 1]])
    tq.xor_game_value(prob_mat, pred_mat, "classical")
    0.75

A quantum strategy for the CHSH game
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The odd cycle game
-------------------


References
----------

