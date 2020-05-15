"""Two-player extended nonlocal game."""
import cvxpy
import numpy as np


class ExtendedNonlocalGame:
    r"""
    Create two-player extended nonlocal game object.

    *Extended nonlocal games* are a superset of nonlocal games in which the
    players share a tripartite state with the referee. In such games, the
    winning conditions for Alice and Bob may depend on outcomes of measurements
    made by the referee, on its part of the shared quantum state, in addition
    to Alice and Bob's answers to the questions sent by the referee.

    Extended nonlocal games were initially defined in [JMRW16]_ and more
    information on these games can be found in [Russo17]_.

    Examples
    ==========

    The BB84 game

    References
    ==========
    .. [JMRW16] Nathaniel Johnston, Rajat Mittal, Vincent Russo, John Watrous
        "Extended non-local games and monogamy-of-entanglement games",
        2016,
        https://arxiv.org/abs/1510.02083

    .. [Russo17] Vincent Russo
        "Extended nonlocal games"
        https://arxiv.org/abs/1704.07375
    """

    def __init__(self, prob_mat: np.ndarray, pred_mat: np.ndarray):
        """
        Construct extended nonlocal game object.

        :param prob_mat:
        :param pred_mat:
        """
        self.prob_mat = prob_mat
        self.pred_mat = pred_mat

    def unentangled_value(self) -> float:
        """
        Calculate the unentangled value of an extended nonlocal game.

        :return:
        """
        dim_x, dim_y, alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape

        max_unent_val = float("-inf")
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                p_win = np.zeros([dim_x, dim_y])
                for x_in in range(alice_in):
                    for y_in in range(bob_in):
                        p_win += (
                            self.prob_mat[x_in, y_in]
                            * self.pred_mat[:, :, a_out, b_out, x_in, y_in]
                        )

                rho = cvxpy.Variable((dim_x, dim_y), hermitian=True)

                objective = cvxpy.Maximize(
                    cvxpy.real(cvxpy.trace(p_win.conj().T @ rho))
                )

                constraints = [cvxpy.trace(rho) == 1, rho >> 0]
                problem = cvxpy.Problem(objective, constraints)
                unent_val = problem.solve()
                max_unent_val = max(max_unent_val, unent_val)
        return max_unent_val

    def nosignaling_value(self) -> float:
        """
        Calculate the no-signaling value of an extended nonlocal game.

        :return:
        """
