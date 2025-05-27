import numpy as np
import numba as nb
from toqito.nonlocal_games.nonlocal_game import NonlocalGame


@nb.njit
def _fast_classical_value(pred_mat: np.ndarray, num_b_out: int, num_b_in: int, pow_arr: np.ndarray):
    r"""Compute the classical winning probability by iterating over all deterministic strategies.

    Expects pred_mat shaped (A_out, A_in, B_out, B_in),
    where B_out == num_b_out and B_in == num_b_in.

    Parameters
    ==========
    pred_mat : np.ndarray
        Weighted predicate matrix of shape (A_out, A_in, B_out, B_in).
    num_b_out : int
        Number of Bob's possible outputs.
    num_b_in : int
        Number of Bob's possible inputs.
    pow_arr : np.ndarray
        Array of powers for decoding Bob's outputs.

    Returns
    =======
    float
        Maximum summed probability over all deterministic strategies.

    Example
    =======
    >>> import numpy as np
    >>> from toqito.nonlocal_games.nonlocal_game import NonlocalGame
    >>> c1 = np.zeros((2, 2)); c2 = np.zeros((2, 2))
    >>> for v1 in range(2):
    ...     for v2 in range(2):
    ...         (c1 if (v1 ^ v2) == 0 else c2)[v1, v2] = 1
    >>> game = NonlocalGame.from_bcs_game([c1, c2])
    >>> game.classical_value_fast()
    0.75
    """
    p_win = 0.0
    total = num_b_out ** num_b_in  # Number of deterministic strategies

    A_out = pred_mat.shape[0]
    A_in = pred_mat.shape[1]

    for i in range(total):
        best_sum = 0.0

        for x in range(A_in):
            best_for_x = 0.0

            for a in range(A_out):
                acc = 0.0

                for y in range(num_b_in):
                    b_q = (i // pow_arr[y]) % num_b_out
                    # Correct axis order: (a, x, b_q, y)
                    acc += pred_mat[a, x, b_q, y]

                if acc > best_for_x:
                    best_for_x = acc

            best_sum += best_for_x

        if best_sum > p_win:
            p_win = best_sum

    return p_win


def classical_value_fast(self) -> float:
    r"""Calculate the classical value using the fast brute-force helper.

    Example
    =======
    >>> import numpy as np
    >>> from toqito.nonlocal_games.nonlocal_game import NonlocalGame
    >>> c1 = np.zeros((2, 2)); c2 = np.zeros((2, 2))
    >>> for v1 in range(2):
    ...     for v2 in range(2):
    ...         (c1 if (v1 ^ v2) == 0 else c2)[v1, v2] = 1
    >>> game = NonlocalGame.from_bcs_game([c1, c2])
    >>> game.classical_value_fast()
    0.75
    """
    pm = np.copy(self.pred_mat)
    pm *= self.prob_mat[np.newaxis, np.newaxis, :, :]

    A_out, B_out, A_in, B_in = pm.shape
    if A_out**A_in < B_out**B_in:
        pm = pm.transpose((1, 0, 3, 2))
        A_out, B_out, A_in, B_in = pm.shape

    pm = pm.transpose((0, 2, 1, 3))

    # 4) Build the power array for decoding
    pow_arr = np.array([B_out ** (B_in - 1 - y) for y in range(B_in)], dtype=np.int64)

    return _fast_classical_value(pm, B_out, B_in, pow_arr)


# Monkey-patch onto NonlocalGame
NonlocalGame.classical_value_fast = classical_value_fast
