"""Compute the classical or quantum value of a two-player nonlocal XOR game."""
import cvxpy
import numpy as np


def xor_game_value(
    prob_mat: np.ndarray,
    pred_mat: np.ndarray,
    strategy: str = "classical",
    tol: float = None,
) -> float:
    """
    Compute the classical or quantum value of a two-player nonlocal XOR game.

    Calculates the optimal probability that Alice and Bob win the game if they
    are allowed to determine a join strategy beforehand, but not allowed to
    communicate during the game itself.

    References:
        [1] Richard Cleve, William Slofstra, Falk Unger, Sarvagya Upadhyay
        "Strong parallel repetition theorem for quantum XOR proof systems",
        https://arxiv.org/abs/quant-ph/0608146

        [2] Richard Cleve, Peter Hoyer, Ben Toner, John Watrous
        "Consequences and limits of nonlocal strategies."
        Proceedings. 19th IEEE Annual Conference on Computational Complexity,
        2004.. IEEE, 2004.
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
