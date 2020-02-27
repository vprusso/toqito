"""Computes the classical or quantum value of a non-local binary XOR game."""
import cvxpy
import numpy as np


def xor_game_value(prob_mat: np.ndarray,
                   pred_mat: np.ndarray,
                   strategy: str = "classical",
                   tol: float = None) -> float:
    """
    Computes the classical or quantum value of a nonlocal binary XOR game.

    Calculates the optimal probability that Alice and Bob win the game if they
    are allowed to determine a join strategy beforehand, but not allowed to
    communicate during the game itself.

    :param prob_mat: A matrix whose (s, t)-entry gives the probability that the
                     referee will give Alice the value s and Bob the value t.
    :param pred_mat: A binary matrix whose (s, t)-entry indicates the winning
                     choice (either 0 or 1) when Alice and Bob receive values s
                     and t from the referee.
    :param strategy: Default is "classical". Either argument "classical" or
                     "quantum" indicating what type of value the game should be
                     computed.
    :param tol: The error tolerance for the value.
    :return: The optimal value that Alice and Bob can win the XOR game using a
             specific type of strategy.
    """
    s, t = prob_mat.shape

    if tol is None:
        tol = np.finfo(float).eps * s**2 * t**2
    if (s, t) != pred_mat.shape:
        raise ValueError("Invalid: The matrices `prob_mat` and `pred_mat` must"
                         " be matrices of the same size.")
    if np.min(np.min(prob_mat)) < -tol:
        raise ValueError("Invalid: The variable `prob_mat` must be a "
                         "probability matrix: its entries must be non-negative.")
    if np.abs(np.sum(np.sum(prob_mat)) - 1) > tol:
        raise ValueError("Invalid: The variable `prob_mat` must be a "
                         "probability matrix: its entries must sum to 1.")

    # Compute the value of the game, depending on which type of value was
    # requested.
    if strategy == "quantum":
        # Use semidefinite programming to compute the value of the game.
        p_var = prob_mat * (1 - 2*pred_mat)
        q_var = np.bmat([[np.zeros((s, s)), p_var],
                         [p_var.conj().T, np.zeros((t, t))]])

        x_var = cvxpy.Variable((s+t, s+t), symmetric=True)
        objective = cvxpy.Maximize(cvxpy.trace(q_var @ x_var))
        constraints = [cvxpy.diag(x_var) == 1, x_var >> 0]
        problem = cvxpy.Problem(objective, constraints)

        return np.real(problem.solve()) / 4 + 1/2

    if strategy == "classical":
        # At worst, out winning probability is 0. Now, try to improve.
        val = 0

        # Find the maximum probability of winning (this is NP-hard, so don't
        # expect an easy way to do it: just loop over all strategies.

        # Loop over Alice's answers
        for a_ans in range(2**s):
            # Loop over Bob's answers:
            for b_ans in range(2**t):
                a_vec = (a_ans >> np.arange(s)) & 1
                b_vec = (b_ans >> np.arange(t)) & 1

                # Now compute the winning probability under this strategy: XOR
                # together Alice's responses and Bob's responses, then check
                # where the XORed value equals the value in the given matrix F.
                # Where the values match, multiply by the probability of
                # getting that pair of questions (i.e., multiply by the
                # probability of getting that pair of questions (i.e., multiply
                # entry-wise by P) and then sum over the rows and columns.
                c_mat = np.mod(
                    np.multiply(a_vec.conj().T.reshape(-1, 1),
                                np.ones((1, t))) +
                    np.multiply(np.ones((s, 1)),
                                b_vec), 2)
                p_win = np.sum(np.sum(np.multiply(c_mat == pred_mat, prob_mat)))

                # Is this strategy better than other ones tried so far?
                val = max(val, p_win)

                # Already optimal? Quit.
                if val >= 1 - tol:
                    return val
        return val
    raise ValueError(f"Strategy {strategy} is not supported.")
