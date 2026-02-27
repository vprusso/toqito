"""Produces the generalized Pauli operator matrices."""

import numpy as np

from toqito.matrices import gen_pauli_x, gen_pauli_z


def gen_pauli(k_1: int, k_2: int, dim: int) -> np.ndarray:
    r"""Produce generalized Pauli operator [@WikiPauliGen].

    Generates a `dim`-by-`dim` unitary operator. More specifically,
    it is the operator \(X^k_1 Z^k_2\), where \(X\) and \(Z\) are
    the "gen_pauli_x" and "gen_pauli_z" operators that naturally generalize the Pauli X and
    Z operators. These matrices span the entire space of
    `dim`-by-`dim` matrices as `k_1` and `k_2` range
    from 0 to `dim-1`, inclusive.

    Note that the generalized Pauli operators are also known by the name of
    "discrete Weyl operators". (Lecture 6: Further Remarks On Measurements And Channels from
    [@Watrous_2011_Lecture_Notes])

    Examples:
        The generalized Pauli operator for `k_1 = 1`, `k_2 = 0`, and
        `dim = 2` is given as the standard Pauli-X matrix

        \[
            G_{1, 0, 2} = \begin{pmatrix}
                             0 & 1 \\
                             1 & 0
                          \end{pmatrix}.
        \]

        This can be obtained in `|toqito⟩` as follows.

        ```python exec="1" source="above"
        from toqito.matrices import gen_pauli

        print(gen_pauli(k_1=1, k_2=0, dim=2))
        ```


        The generalized Pauli matrix `k_1 = 1`, `k_2 = 1`, and
        `dim = 2` is given as the standard Pauli-Y matrix

        \[
            G_{1, 1, 2} = \begin{pmatrix}
                            0 & -1 \\
                            1 & 0
                          \end{pmatrix}.
        \]

        This can be obtained in `|toqito⟩` as follows.

        ```python exec="1" source="above"
        from toqito.matrices import gen_pauli

        print(gen_pauli(k_1=1, k_2=1, dim=2))
        ```

    Args:
        k_1: (a non-negative integer from 0 to `dim-1` inclusive).
        k_2: (a non-negative integer from 0 to `dim-1` inclusive).
        dim: (a positive integer indicating the dimension).

    Returns:
        A generalized Pauli operator.

    """
    gpx_val = gen_pauli_x(dim)
    gpz_val = gen_pauli_z(dim)

    gen_pauli_w = np.linalg.matrix_power(gpx_val, k_1) @ np.linalg.matrix_power(gpz_val, k_2)

    return gen_pauli_w
