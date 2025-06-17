"""Chessboard state represent the state of a chessboard used in quantum chess.

In a quantum chessboard, each chess piece is quantum having a superposition of channel states, giving rise to a unique
chess piece.
"""

import numpy as np


def chessboard(mat_params: list[float], s_param: float = None, t_param: float = None) -> np.ndarray:
    r"""Produce a chessboard state :footcite:`Bruß_2000_Construction`.

    Generates the chessboard state defined in :footcite:`Bruß_2000_Construction`. Note that, for certain choices of
    :code:`s_param` and :code:`t_param`, this state will not have positive partial transpose, and
    thus may not be bound entangled.

    Examples
    ==========

    The standard chessboard state can be invoked using :code:`|toqito⟩` as

    .. jupyter-execute::

      from toqito.states import chessboard
      chessboard([1, 2, 3, 4, 5, 6], 7, 8)


    References
    ==========
    .. footbibliography::



    :param mat_params: Parameters of the chessboard state as defined in :footcite:`Bruß_2000_Construction`.
    :param s_param: Default is :code:`np.conj(mat_params[2]) / np.conj(mat_params[5])`.
    :param t_param: Default is :code:`t_param = mat_params[0] * mat_params[3] / mat_params[4]`.
    :return: A chessboard state.

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
