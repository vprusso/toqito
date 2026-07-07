"""Produces the generalized Pauli operator matrices."""

import numpy as np

from toqito.matrices import gen_pauli_z


def gen_pauli(k_1: int, k_2: int, dim: int) -> np.ndarray:
    r"""Produce generalized Pauli operator [@wikipediageneralizedpauli].

    Generates a `dim`-by-`dim` unitary operator. More specifically,
    it is the operator \(X^k_1 Z^k_2\), where \(X\) and \(Z\) are
    the "gen_pauli_x" and "gen_pauli_z" operators that naturally generalize the Pauli X and
    Z operators. These matrices span the entire space of
    `dim`-by-`dim` matrices as `k_1` and `k_2` range
    from 0 to `dim-1`, inclusive.

    Note that the generalized Pauli operators are also known by the name of
    "discrete Weyl operators". (Lecture 6: Further Remarks On Measurements And Channels from
    [@watrous2011theory])

    Args:
        k_1: (a non-negative integer from 0 to `dim-1` inclusive).
        k_2: (a non-negative integer from 0 to `dim-1` inclusive).
        dim: (a positive integer indicating the dimension).

    Returns:
        A generalized Pauli operator.

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

        ```python exec="1" source="above" result="text"
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

        ```python exec="1" source="above" result="text"
        from toqito.matrices import gen_pauli

        print(gen_pauli(k_1=1, k_2=1, dim=2))
        ```

    """
    gpz_val = gen_pauli_z(dim)

    # X^{k_1} is a cyclic column shift of the identity, and Z^{k_2} is the entrywise power of the
    # diagonal gen_pauli_z operator. Right-multiplication by the diagonal Z^{k_2} scales column j,
    # so the product X^{k_1} Z^{k_2} is the shifted identity times the diagonal broadcast over rows.
    z_diag = np.diag(gpz_val) ** k_2
    gen_pauli_w = np.roll(np.identity(dim), -k_1 % dim, axis=1) * z_diag

    return gen_pauli_w
