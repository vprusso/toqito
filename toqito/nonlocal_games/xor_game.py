"""Two-player XOR game."""
import cvxpy
import numpy as np

from toqito.nonlocal_games.nonlocal_game import NonlocalGame


class XORGame:
    r"""Create two-player XOR game object.

    Calculates the optimal probability that Alice and Bob win the game if they
    are allowed to determine a join strategy beforehand, but not allowed to
    communicate during the game itself.

    The quantum value of an XOR game can be solved via the semidefinite program
    from :cite:`Cleve_2010_Consequences`.

    This function is adapted from the QETLAB package.

    A tutorial is available in the documentation. Go to :ref:`ref-label-xor-quantum-value-tutorial`.

    Examples
    ==========

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
    >>> chsh.quantum_value()
    0.8535533885683664
    >>>
    >>> chsh.classical_value()
    0.75

    The odd cycle game

    The odd cycle game is another XOR game :cite:`Cleve_2010_Consequences`. For this game, we can
    specify the probability and predicate matrices as follows.

    >>> prob_mat = np.array(
    >>> [
    >>>     [0.1, 0.1, 0, 0, 0],
    >>>     [0, 0.1, 0.1, 0, 0],
    >>>     [0, 0, 0.1, 0.1, 0],
    >>>     [0, 0, 0, 0.1, 0.1],
    >>>     [0.1, 0, 0, 0, 0.1],
    >>> ]
    >>> )
    >>> pred_mat = np.array(
    >>> [
    >>>     [0, 1, 0, 0, 0],
    >>>     [0, 0, 1, 0, 0],
    >>>     [0, 0, 0, 1, 0],
    >>>     [0, 0, 0, 0, 1],
    >>>     [1, 0, 0, 0, 0],
    >>> ]
    >>> )

    In :code:`toqito`, we can calculate both the quantum and classical value of
    the odd cycle game as follows.

    >>> import numpy as np
    >>> from toqito.nonlocal_games.xor_game import XORGame
    >>> odd_cycle = XORGame(prob_mat, pred_mat)
    >>> odd_cycle.quantum_value()
    0.9755282544736033
    >>> odd_cycle.classical_value()
    0.9

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    """

    def __init__(
        self,
        prob_mat: np.ndarray,
        pred_mat: np.ndarray,
        reps: int = 1,
        tol: float = None,
    ) -> None:
        """Construct XOR game object.

        :raises ValueError: If dimension of probability matrix is not equal to
                                                dimension of predicate matrix.
        :param prob_mat: A matrix whose (q_0, q_1)-entry gives the probability that
                                 the referee will give Alice the value `q_0` and Bob the
                                 value `q_1`.
        :param pred_mat: A binary matrix whose (q_0, q_1)-entry indicates the
                                 winning choice (either 0 or 1) when Alice and Bob receive
                                 values `q_0` and `q_1` from the referee.
        :param tol: The error tolerance for the value.
        """
        self.prob_mat = prob_mat
        self.pred_mat = pred_mat
        self.reps = reps

        q_0, q_1 = self.prob_mat.shape
        if tol is None:
            self.tol = np.finfo(float).eps * q_0 ** 2 * q_1 ** 2
        else:
            self.tol = tol

        # Perform some basic error checking to ensure the probability and
        # predicate matrices are well-defined.
        if (q_0, q_1) != self.pred_mat.shape:
            raise ValueError(
                "Invalid: The matrices `prob_mat` and `pred_mat` must"
                " be matrices of the same size."
            )
        if -np.min(np.min(self.prob_mat)) > self.tol:
            raise ValueError(
                "Invalid: The variable `prob_mat` must be a "
                "probability matrix: its entries must be "
                "non-negative."
            )
        if np.abs(np.sum(np.sum(self.prob_mat)) - 1) > self.tol:
            raise ValueError(
                "Invalid: The variable `prob_mat` must be a "
                "probability matrix: its entries must sum to 1."
            )

    def quantum_value(self) -> float:
        r"""Compute the quantum value of the XOR game.

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
        """
        alice_in, bob_in = self.prob_mat.shape
        d_mat = np.zeros([alice_in, bob_in])

        for x_alice in range(alice_in):
            for y_bob in range(bob_in):
                d_mat[x_alice, y_bob] = self.prob_mat[x_alice, y_bob] * (-1) ** (
                    self.pred_mat[x_alice, y_bob]
                )

        u_vec = cvxpy.Variable(alice_in, complex=False)
        v_vec = cvxpy.Variable(bob_in, complex=False)

        objective = cvxpy.Minimize(cvxpy.sum(u_vec) + cvxpy.sum(v_vec))
        constraints = [
            cvxpy.bmat(
                [
                    [cvxpy.diag(u_vec), -d_mat],
                    [np.negative(d_mat.conj().T), cvxpy.diag(v_vec)],
                ]
            )
            >> 0
        ]

        problem = cvxpy.Problem(objective, constraints)
        problem.solve()

        if self.reps == 1:
            return np.real(problem.value) / 4 + 1 / 2
        # It holds from (https://arxiv.org/abs/quant-ph/0608146) that the
        # quantum value of any XOR game obeys strong parallel repetition. That
        # is, it holds that:
        #   \omega^*(G^{^n}) = \omega^*(G)^n,
        # where G^{^n} denotes playing the game G n-times.
        return (np.real(problem.value) / 4 + 1 / 2) ** self.reps

    def classical_value(self) -> float:
        """Compute the classical value of the XOR game.

        :return: A value between [0, 1] representing the classical value.
        """
        return self.to_nonlocal_game().classical_value()

    def nonsignaling_value(self) -> float:
        """Compute the nonsignaling value of an XOR game.

        Here, the exising function in the :code:`NonlocalGame` class is called.

        :return: A value between [0, 1] representing the nonsignaling value.
        """
        return self.to_nonlocal_game().nonsignaling_value()

    def to_nonlocal_game(self) -> NonlocalGame:
        """Given an XOR game, compute a predicate matrix representing the more generic :code:`NonlocalGame` equivalent.

        :return: A :code:`NonlocalGame` object equivalent to the XOR game.
        """
        q_0, q_1 = self.prob_mat.shape
        xor_pred_mat = self.pred_mat
        nlg_pred_mat = np.ndarray((2, 2, q_0, q_1))

        for a in range(2):
            for b in range(2):
                for x in range(q_0):
                    for y in range(q_1):
                        nlg_pred_mat[a, b, x, y] = xor_pred_mat[x, y] == a ^ b

        return NonlocalGame(self.prob_mat, nlg_pred_mat, reps=self.reps)
