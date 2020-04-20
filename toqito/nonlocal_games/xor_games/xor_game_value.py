"""Compute the classical or quantum value of a two-player nonlocal XOR game."""
import cvxpy
import numpy as np


def xor_game_value(
    prob_mat: np.ndarray,
    pred_mat: np.ndarray,
    strategy: str = "classical",
    tol: float = None,
) -> float:
    r"""
    Compute the classical or quantum value of a two-player nonlocal XOR game.

    Calculates the optimal probability that Alice and Bob win the game if they
    are allowed to determine a join strategy beforehand, but not allowed to
    communicate during the game itself.

    The quantum value of an XOR game can be solved via the semidefinite program
    from [2]_.

    This function is adapted from the QETLAB package.

    Examples
    ==========

    The CHSH game

    The CHSH game is a two-player nonlocal game with the following probability
    distribution and question and answer sets [1]_.

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

    Recall that :math`\oplus` refers to the XOR operation.

    The optimal quantum value of CHSH is :math:`\cos(\pi/8)^2 \approx 0.8536`
    where the optimal classical value is :math:`3/4`.

    In order to specify the CHSH game, we can define the probability matrix and
    predicate matrix for the CHSH game as `numpy` arrays as follows.

    >>> prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
    >>> pred_mat = np.array([[0, 0], [0, 1]])

    In `toqito`, we can calculate both the quantum and classical value of the
    CHSH game as follows.

    >>> import numpy as np
    >>> from toqito.nonlocal_games.xor_games.xor_game_value import xor_game_value
    >>> xor_game_value(prob_mat, pred_mat, "quantum")
    0.8535533885683664
    >>>
    >>> xor_game_value(prob_mat, pred_mat, "classical")
    0.75

    The odd cycle game

    The odd cycle game is another XOR game [2]_. For this game, we can specify
    the probability and predicate matrices as follows.

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

    In `toqito`, we can calculate both the quantum and classical value of the
    odd cycle game as follows.

    >>> import numpy as np
    >>> from toqito.nonlocal_games.xor_games.xor_game_value import xor_game_value
    >>> xor_game_value(prob_mat, pred_mat, "quantum")
    0.9755282544736033
    >>> xor_game_value(prob_mat, pred_mat, "classical")
    0.9

    References
    ==========
    .. [1] Richard Cleve, William Slofstra, Falk Unger, Sarvagya Upadhyay
        "Strong parallel repetition theorem for quantum XOR proof systems",
        https://arxiv.org/abs/quant-ph/0608146

    .. [2] Richard Cleve, Peter Hoyer, Ben Toner, John Watrous
        "Consequences and limits of nonlocal strategies."
        Proceedings. 19th IEEE Annual Conference on Computational Complexity,
        IEEE, 2004.
        https://arxiv.org/abs/quant-ph/0404076

    :param prob_mat: A matrix whose (q_0, q_1)-entry gives the probability that
                     the referee will give Alice the value `q_0` and Bob the
                     value `q_1`.
    :param pred_mat: A binary matrix whose (q_0, q_1)-entry indicates the
                     winning choice (either 0 or 1) when Alice and Bob receive
                     values `q_0` and `q_1` from the referee.
    :param strategy: Default is "classical". Either argument "classical" or
                     "quantum" indicating what type of value the game should be
                     computed.
    :param tol: The error tolerance for the value.
    :return: The optimal value that Alice and Bob can win the XOR game using a
             specific type of strategy.
    """
    q_0, q_1 = prob_mat.shape

    if tol is None:
        tol = np.finfo(float).eps * q_0 ** 2 * q_1 ** 2
    if (q_0, q_1) != pred_mat.shape:
        raise ValueError(
            "Invalid: The matrices `prob_mat` and `pred_mat` must"
            " be matrices of the same size."
        )
    if -np.min(np.min(prob_mat)) > tol:
        raise ValueError(
            "Invalid: The variable `prob_mat` must be a "
            "probability matrix: its entries must be "
            "non-negative."
        )
    if np.abs(np.sum(np.sum(prob_mat)) - 1) > tol:
        raise ValueError(
            "Invalid: The variable `prob_mat` must be a "
            "probability matrix: its entries must sum to 1."
        )

    # Compute the value of the game, depending on which type of value was
    # requested.
    if strategy == "quantum":
        # Use semidefinite programming to compute the value of the game.
        p_var = prob_mat * (1 - 2 * pred_mat)
        q_var = np.bmat(
            [[np.zeros((q_0, q_0)), p_var], [p_var.conj().T, np.zeros((q_1, q_1))]]
        )

        x_var = cvxpy.Variable((q_0 + q_1, q_0 + q_1), symmetric=True)
        objective = cvxpy.Maximize(cvxpy.trace(q_var @ x_var))
        constraints = [cvxpy.diag(x_var) == 1, x_var >> 0]
        problem = cvxpy.Problem(objective, constraints)

        return np.real(problem.solve()) / 4 + 1 / 2

    if strategy == "classical":
        # At worst, out winning probability is 0. Now, try to improve.
        val = 0

        # Find the maximum probability of winning (this is NP-hard, so don't
        # expect an easy way to do it: just loop over all strategies.

        # Loop over Alice's answers
        for a_ans in range(2 ** q_0):
            # Loop over Bob's answers:
            for b_ans in range(2 ** q_1):
                a_vec = (a_ans >> np.arange(q_0)) & 1
                b_vec = (b_ans >> np.arange(q_1)) & 1

                # Now compute the winning probability under this strategy: XOR
                # together Alice's responses and Bob's responses, then check
                # where the XORed value equals the value in the given matrix F.
                # Where the values match, multiply by the probability of
                # getting that pair of questions (i.e., multiply by the
                # probability of getting that pair of questions (i.e., multiply
                # entry-wise by P) and then sum over the rows and columns.
                c_mat = np.mod(
                    np.multiply(a_vec.conj().T.reshape(-1, 1), np.ones((1, q_1)))
                    + np.multiply(np.ones((q_0, 1)), b_vec),
                    2,
                )
                p_win = np.sum(np.sum(np.multiply(c_mat == pred_mat, prob_mat)))

                # Is this strategy better than other ones tried so far?
                val = max(val, p_win)

                # Already optimal? Quit.
                if val >= 1 - tol:
                    return val
        return val
    raise ValueError(f"Strategy {strategy} is not supported.")
