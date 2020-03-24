"""Produces a chessboard state."""
from typing import List
import numpy as np


def chessboard(mat_params: List[float],
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
        s_param = np.conj(mat_params[2])/np.conj(mat_params[5])
    if t_param is None:
        t_param = mat_params[0] * mat_params[3]/mat_params[4]

    v_1 = np.array([[mat_params[4], 0, s_param, 0, mat_params[5], 0, 0, 0, 0]])

    v_2 = np.array([[0, mat_params[0], 0, mat_params[1],
                     0, mat_params[2], 0, 0, 0]])

    v_3 = np.array([[np.conj(mat_params[5]), 0, 0, 0,
                     -np.conj(mat_params[4]), 0, t_param, 0, 0]])

    v_4 = np.array([[0, np.conj(mat_params[1]), 0,
                     -np.conj(mat_params[0]), 0, 0, 0, mat_params[3], 0]])

    rho = v_1.conj().T * v_1 + v_2.conj().T * v_2 + \
        v_3.conj().T * v_3 + v_4.conj().T * v_4
    return rho/np.trace(rho)
