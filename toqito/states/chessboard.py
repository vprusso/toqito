"""Chessboard state."""
from typing import List

import numpy as np


def chessboard(mat_params: List[float], s_param: float = None, t_param: float = None) -> np.ndarray:
    r"""
    Produce a chessboard state [BP00]_.

    Generates the chessboard state defined in [BP00]_. Note that, for certain choices of
    :code:`s_param` and :code:`t_param`, this state will not have positive partial transpose, and
    thus may not be bound entangled.

    Examples
    ==========

    The standard chessboard state can be invoked using :code:`toqito` as

    >>> from toqito.states import chessboard
    >>> chessboard([1, 2, 3, 4, 5, 6], 7, 8)
    [[ 0.22592593,  0.        ,  0.12962963,  0.        ,  0.        ,
       0.        ,  0.17777778,  0.        ,  0.        ],
     [ 0.        ,  0.01851852,  0.        ,  0.        ,  0.        ,
       0.01111111,  0.        ,  0.02962963,  0.        ],
     [ 0.12962963,  0.        ,  0.18148148,  0.        ,  0.15555556,
       0.        ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.01851852,  0.        ,
       0.02222222,  0.        , -0.01481481,  0.        ],
     [ 0.        ,  0.        ,  0.15555556,  0.        ,  0.22592593,
       0.        , -0.14814815,  0.        ,  0.        ],
     [ 0.        ,  0.01111111,  0.        ,  0.02222222,  0.        ,
       0.03333333,  0.        ,  0.        ,  0.        ],
     [ 0.17777778,  0.        ,  0.        ,  0.        , -0.14814815,
       0.        ,  0.23703704,  0.        ,  0.        ],
     [ 0.        ,  0.02962963,  0.        , -0.01481481,  0.        ,
       0.        ,  0.        ,  0.05925926,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       0.        ,  0.        ,  0.        ,  0.        ]]

    References
    ==========
    .. [BP00] Three qubits can be entangled in two inequivalent ways.
        D. Bruss and A. Peres
        Phys. Rev. A, 61:30301(R), 2000
        arXiv: 991.1056

    :param mat_params:
    :param s_param:
    :param t_param:
    :return: A chessboard state.
    """
    if s_param is None:
        s_param = np.conj(mat_params[2]) / np.conj(mat_params[5])
    if t_param is None:
        t_param = mat_params[0] * mat_params[3] / mat_params[4]

    v_1 = np.array([[mat_params[4], 0, s_param, 0, mat_params[5], 0, 0, 0, 0]])

    v_2 = np.array([[0, mat_params[0], 0, mat_params[1], 0, mat_params[2], 0, 0, 0]])

    v_3 = np.array([[np.conj(mat_params[5]), 0, 0, 0, -np.conj(mat_params[4]), 0, t_param, 0, 0]])

    v_4 = np.array(
        [
            [
                0,
                np.conj(mat_params[1]),
                0,
                -np.conj(mat_params[0]),
                0,
                0,
                0,
                mat_params[3],
                0,
            ]
        ]
    )

    rho = v_1.conj().T * v_1 + v_2.conj().T * v_2 + v_3.conj().T * v_3 + v_4.conj().T * v_4
    return rho / np.trace(rho)
