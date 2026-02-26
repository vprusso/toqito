"""Chessboard state represent the state of a chessboard used in quantum chess.

In a quantum chessboard, each chess piece is quantum having a superposition of channel states, giving rise to a unique
chess piece.
"""

import numpy as np


def chessboard(mat_params: list[float], s_param: float | None = None, t_param: float | None = None) -> np.ndarray:
    r"""Produce a chessboard state [@Bruß_2000_Construction].

    Generates the chessboard state defined in [@Bruß_2000_Construction]. Note that, for certain choices of
    `s_param` and `t_param`, this state will not have positive partial transpose, and
    thus may not be bound entangled.

    Examples:
    The standard chessboard state can be invoked using `|toqito⟩` as

    ```python exec="1" source="above"
    from toqito.states import chessboard
    print(chessboard([1, 2, 3, 4, 5, 6], 7, 8))
    ```

    Args:
        mat_params: Parameters of the chessboard state as defined in [@Bruß_2000_Construction].
        s_param: Default is `np.conj(mat_params[2]) / np.conj(mat_params[5])`.
        t_param: Default is `t_param = mat_params[0] * mat_params[3] / mat_params[4]`.

    Returns:
        A chessboard state.

    """
    if s_param is None:
        s_param = np.conj(mat_params[2]) / np.conj(mat_params[5])
    if t_param is None:
        t_param = mat_params[0] * mat_params[3] / mat_params[4]

    v_1 = np.array([[mat_params[4], 0, s_param, 0, mat_params[5], 0, 0, 0, 0]])
    v_2 = np.array([[0, mat_params[0], 0, mat_params[1], 0, mat_params[2], 0, 0, 0]])
    v_3 = np.array([[np.conj(mat_params[5]), 0, 0, 0, -np.conj(mat_params[4]), 0, t_param, 0, 0]])
    v_4 = np.array([[0, np.conj(mat_params[1]), 0, -np.conj(mat_params[0]), 0, 0, 0, mat_params[3], 0]])
    rho = v_1.conj().T @ v_1 + v_2.conj().T @ v_2 + v_3.conj().T @ v_3 + v_4.conj().T @ v_4
    return rho / np.trace(rho)
