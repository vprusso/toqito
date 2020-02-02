"""Produces a chessboard state."""
import numpy as np


def chessboard_state(a_param: float,
                     b_param: float,
                     c_param: float,
                     d_param: float,
                     m_param: float,
                     n_param: float,
                     s_param: float = None,
                     t_param: float = None) -> np.ndarray:
    """
    Produce a chessboard state.

    Generates the chessboard state defined in [1]. Note that, for certain
    choices of S and T, this state will not have positive partial transpose,
    and thus may not be bound entangled.

    References:
    [1] Three qubits can be entangled in two inequivalent ways.
        D. Bruss and A. Peres
        Phys. Rev. A, 61:30301(R), 2000
        arXiv: 991.1056
    """
    if s_param is None:
        s_param = np.conj(c_param)/np.conj(n_param)
    if t_param is None:
        t_param = a_param * d_param/m_param

    v_1 = np.array([[m_param, 0, s_param, 0, n_param, 0, 0, 0, 0]])
    v_2 = np.array([[0, a_param, 0, b_param, 0, c_param, 0, 0, 0]])
    v_3 = np.array([[np.conj(n_param), 0, 0, 0,
                     -np.conj(m_param), 0, t_param, 0, 0]])
    v_4 = np.array([[0, np.conj(b_param), 0,
                     -np.conj(a_param), 0, 0, 0, d_param, 0]])

    rho = v_1.conj().T * v_1 + v_2.conj().T * v_2 + \
        v_3.conj().T * v_3 + v_4.conj().T * v_4
    return rho/np.trace(rho)
