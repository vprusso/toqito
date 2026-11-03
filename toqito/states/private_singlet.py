"""Private-singlet bound entangled state."""

import numpy as np

from toqito.matrices import fourier
from toqito.perms import permute_systems
from toqito.states import basis


def private_singlet(shield_dim: int) -> np.ndarray:
    r"""Produce the private-singlet bound entangled state [@badziag2014bound].

    This is the first family of PPT singlet-like states, \(\rho_{F1}\), of [@pal2021bound], originally
    constructed in [@badziag2014bound] from private states. It lives on
    \(\mathbb{C}^2 \otimes \mathbb{C}^2 \otimes \mathbb{C}^d \otimes \mathbb{C}^d\), a two-qubit pair
    \(AB\) and a shield pair \(A'B'\) of local dimension \(d\). Writing
    \(|ij, kl\rangle = |i\rangle_A |j\rangle_B |k\rangle_{A'} |l\rangle_{B'}\), with weights

    \[
        p_1 = \frac{\sqrt{d}}{1 + \sqrt{d}}, \qquad p_2 = 1 - p_1,
    \]

    and the Fourier matrix \(u_{ij} = \omega^{ij}/\sqrt{d}\), \(\omega = e^{2\pi i/d}\), the state is the sum
    of the two branches

    \[
        \frac{p_1}{2d^2} \sum_{i,j} \big(|00,ij\rangle\langle 00,ij| + |11,ij\rangle\langle 11,ij|\big)
        + \frac{p_1}{2d\sqrt{d}} \sum_{i,j} \big(u_{ij} |00,ij\rangle\langle 11,ji| + \text{h.c.}\big)
    \]

    and

    \[
        \frac{p_2}{2d} \sum_{i} \big(|01,ii\rangle\langle 01,ii| + |10,ii\rangle\langle 10,ii|\big)
        + \frac{p_2}{2d} \sum_{i,j} \big(u_{ij} |01,ii\rangle\langle 10,jj| + \text{h.c.}\big).
    \]

    The returned matrix is reordered to \(AA'BB'\), so the cut between Alice \((A, A')\) and Bob
    \((B, B')\), each of dimension \(2d\), can be tested directly with `dim=[2 * shield_dim, 2 * shield_dim]`.
    The state is invariant under partial transposition across that cut and entangled for every
    \(d \geq 2\). The construction matches the `BES_private` routine of QUBIT4MATLAB
    [@toth2008qubit4matlab].

    Args:
        shield_dim: The shield dimension \(d \geq 2\).

    Returns:
        The private-singlet state on \(\mathbb{C}^{2d} \otimes \mathbb{C}^{2d}\) in \(AA'BB'\) order.

    Raises:
        ValueError: If `shield_dim` is smaller than 2.

    Examples:
        For \(d = 2\) the state acts on \(\mathbb{C}^4 \otimes \mathbb{C}^4\) and is PPT.

        ```python exec="1" source="above" result="text"
        from toqito.states import private_singlet
        from toqito.state_props import is_ppt
        rho = private_singlet(2)
        print(rho.shape, is_ppt(rho, dim=[4, 4]))
        ```

    """
    if shield_dim < 2:
        raise ValueError("The shield dimension must be at least 2.")
    d = shield_dim
    p_1 = np.sqrt(d) / (1 + np.sqrt(d))
    p_2 = 1 - p_1
    u_mat = fourier(d)

    def ket(i: int, j: int, k: int, m: int) -> np.ndarray:
        return np.kron(np.kron(np.kron(basis(2, i), basis(2, j)), basis(d, k)), basis(d, m))

    rho = np.zeros((4 * d * d, 4 * d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            ket_00 = ket(0, 0, i, j)
            ket_11 = ket(1, 1, j, i)
            rho += p_1 / (2 * d * d) * (ket_00 @ ket_00.T + ket(1, 1, i, j) @ ket(1, 1, i, j).T)
            cross = u_mat[i, j] * ket_00 @ ket_11.T
            rho += p_1 / (2 * d * np.sqrt(d)) * (cross + cross.conj().T)

    for i in range(d):
        ket_01 = ket(0, 1, i, i)
        ket_10 = ket(1, 0, i, i)
        rho += p_2 / (2 * d) * (ket_01 @ ket_01.T + ket_10 @ ket_10.T)
        for j in range(d):
            cross = u_mat[i, j] * ket_01 @ ket(1, 0, j, j).T
            rho += p_2 / (2 * d) * (cross + cross.conj().T)

    return permute_systems(rho, [0, 2, 1, 3], dim=[2, 2, d, d])
