"""Chessboard states are a six-parameter family of two-qutrit bound entangled states.

They are named for their matrix: with the ninth row and column empty, the remaining 8-by-8 block has the zero
pattern of a chessboard.
"""

import numpy as np


def chessboard(mat_params: list[float], s_param: float | None = None, t_param: float | None = None) -> np.ndarray:
    r"""Produce a chessboard state [@bruss2000construction].

    The chessboard state with real parameters \(a, b, c, d, m, n\) is the normalized sum of four rank-one
    projectors \(\rho \propto \sum_{i=1}^4 |V_i\rangle\langle V_i|\), where, writing each vector of
    \(\mathbb{C}^3 \otimes \mathbb{C}^3\) as its three blocks of three coefficients,

    \[
        \begin{aligned}
            |V_1\rangle &= (m, 0, s;\ 0, n, 0;\ 0, 0, 0), \\
            |V_2\rangle &= (0, a, 0;\ b, 0, c;\ 0, 0, 0), \\
            |V_3\rangle &= (n, 0, 0;\ 0, -m, 0;\ t, 0, 0), \\
            |V_4\rangle &= (0, b, 0;\ -a, 0, 0;\ 0, d, 0).
        \end{aligned}
    \]

    With \(s = ac/n\) and \(t = ad/m\), the choice made in [@bruss2000construction], the state is invariant
    under partial transposition and hence PPT, while the range criterion shows it is entangled. These are the
    defaults for `s_param` and `t_param`; other choices generally give a state with negative partial
    transpose. The member with \(m = n = b = -3/5\), \(a = 3/5\) and \(c = -d = 6/5\) is the extremal PPT
    entangled state studied in [@imai2021bound].

    Args:
        mat_params: The real parameters \([a, b, c, d, m, n]\).
        s_param: Defaults to \(ac/n\).
        t_param: Defaults to \(ad/m\).

    Returns:
        A chessboard state.

    Examples:
        With the default `s_param` and `t_param` the state is PPT.

        ```python exec="1" source="above" result="text"
        from toqito.states import chessboard
        from toqito.state_props import is_ppt
        rho = chessboard([1, 2, 3, 4, 5, 6])
        print(is_ppt(rho, dim=[3, 3]))
        ```

    """
    a_param, b_param, c_param, d_param, m_param, n_param = mat_params
    if s_param is None:
        s_param = a_param * c_param / n_param
    if t_param is None:
        t_param = a_param * d_param / m_param

    v_1 = np.array([[m_param, 0, s_param, 0, n_param, 0, 0, 0, 0]])
    v_2 = np.array([[0, a_param, 0, b_param, 0, c_param, 0, 0, 0]])
    v_3 = np.array([[np.conj(n_param), 0, 0, 0, -np.conj(m_param), 0, t_param, 0, 0]])
    v_4 = np.array([[0, np.conj(b_param), 0, -np.conj(a_param), 0, 0, 0, d_param, 0]])
    rho = v_1.conj().T @ v_1 + v_2.conj().T @ v_2 + v_3.conj().T @ v_3 + v_4.conj().T @ v_4
    return rho / np.trace(rho)
