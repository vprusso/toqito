"""Two-player extended nonlocal game."""
import cvxpy
import numpy as np

from toqito.state_ops import tensor
from toqito.helper import update_odometer


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

    Let :math:`\Sigma_A = \Sigma_B = \Gamma_A = \Gamma_B = \{0, 1\}`, define

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

    def __init__(
        self, prob_mat: np.ndarray, pred_mat: np.ndarray, reps: int = 1
    ) -> None:
        """
        Construct extended nonlocal game object.

        :param prob_mat:
        :param pred_mat:
        :param reps:
        """
        if reps == 1:
            self.prob_mat = prob_mat
            self.pred_mat = pred_mat
            self.reps = reps

        else:
            (
                dim_x,
                dim_y,
                num_alice_out,
                num_bob_out,
                num_alice_in,
                num_bob_in,
            ) = pred_mat.shape
            self.prob_mat = tensor(prob_mat, reps)

            pred_mat2 = np.zeros(
                (
                    dim_x ** reps,
                    dim_y ** reps,
                    num_alice_out ** reps,
                    num_bob_out ** reps,
                    num_alice_in ** reps,
                    num_bob_in ** reps,
                )
            )
            i_ind = np.zeros(reps, dtype=int)
            j_ind = np.zeros(reps, dtype=int)
            for i in range(num_alice_in ** reps):
                for j in range(num_bob_in ** reps):
                    to_tensor = np.empty(
                        [reps, dim_x, dim_y, num_alice_out, num_bob_out]
                    )
                    for k in range(reps - 1, -1, -1):
                        to_tensor[k] = pred_mat[:, :, :, :, i_ind[k], j_ind[k]]
                    pred_mat2[:, :, :, :, i, j] = tensor(to_tensor)
                    j_ind = update_odometer(j_ind, num_bob_in * np.ones(reps))
                i_ind = update_odometer(i_ind, num_alice_in * np.ones(reps))
            self.pred_mat = pred_mat2
            self.reps = reps

    def unentangled_value(self) -> float:
        r"""
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
        """
        dim_x, dim_y, alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape

        max_unent_val = float("-inf")
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                p_win = np.zeros([dim_x, dim_y], dtype=complex)
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

    def nonsignaling_value(self) -> float:
        """
        Calculate the non-signaling value of an extended nonlocal game.

        :return: The non-signaling value of the extended nonlocal game.
        """
