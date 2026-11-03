"""Orthogonal-singlet bound entangled state."""

import numpy as np

from toqito.perms import permute_systems
from toqito.states import basis


def orthogonal_singlet(shield_dim: int) -> np.ndarray:
    r"""Produce the orthogonal-singlet bound entangled state \(\rho_{F2}\) [@pal2021bound].

    This is the second family of PPT singlet-like states of [@pal2021bound], obtained from the states
    found numerically in [@toth2018quantum]. Like the private singlet it lives on
    \(\mathbb{C}^2 \otimes \mathbb{C}^2 \otimes \mathbb{C}^d \otimes \mathbb{C}^d\), a two-qubit pair
    \(AB\) and a shield pair \(A'B'\), with weights

    \[
        p_1 = \frac{\sqrt{d}}{1 + \sqrt{d}}, \qquad p_2 = 1 - p_1,
    \]

    but the Fourier phases are replaced by a family of \(d\) real orthogonal \(d \times d\) matrices
    \(Q^k\). Writing \(|ij, kl\rangle = |i\rangle_A |j\rangle_B |k\rangle_{A'} |l\rangle_{B'}\),

    \[
        \rho = \frac{p_1}{d^2} \sum_{i,j} |z_{ij}\rangle\langle z_{ij}|
        + \frac{p_2}{2d} \sum_{k} |01\rangle\langle 01|_{AB} \otimes |s_k\rangle\langle s_k|_{A'B'}
        + \frac{p_2}{2d} \sum_{i} |10, ii\rangle\langle 10, ii|,
    \]

    where \(|z_{ij}\rangle = \frac{1}{\sqrt{2}} \big( |00, ij\rangle + \sum_k Q^j_{ik} |11, jk\rangle \big)\).
    For \(d = 3\) the matrices \(Q^k\) are the reflections with \(\varphi = 2\pi(k+1)/3\),

    \[
        Q^k = \begin{pmatrix} \cos\varphi & \sin\varphi & 0 \\ \sin\varphi & -\cos\varphi & 0 \\
        0 & 0 & 1 \end{pmatrix},
    \]

    with \(|s_0\rangle = (|00\rangle + |11\rangle)/\sqrt{2}\), \(|s_1\rangle = (|01\rangle - |10\rangle)/\sqrt{2}\)
    and \(|s_2\rangle = |22\rangle\). For \(d = 2^n\), \(Q^k\) is the tensor product of Pauli \(X\) and
    identity factors selected by the bits of \(k\), and \(|s_k\rangle = \frac{1}{\sqrt{d}} \sum_{ij} Q^k_{ij}
    |ij\rangle\).

    The returned matrix is reordered to \(AA'BB'\), so the cut between Alice \((A, A')\) and Bob
    \((B, B')\), each of dimension \(2d\), can be tested directly with `dim=[2 * shield_dim, 2 * shield_dim]`.
    The state is PPT but, unlike the private singlet, not invariant under partial transposition. The
    construction is a port of the `BES_metro` routine of QUBIT4MATLAB [@toth2008qubit4matlab].

    Args:
        shield_dim: The shield dimension \(d\), which must be 3 or a power of 2.

    Returns:
        The orthogonal-singlet state on \(\mathbb{C}^{2d} \otimes \mathbb{C}^{2d}\) in \(AA'BB'\) order.

    Raises:
        ValueError: If `shield_dim` is neither 3 nor a power of 2.

    Examples:
        For \(d = 2\) the state acts on \(\mathbb{C}^4 \otimes \mathbb{C}^4\) and is PPT.

        ```python exec="1" source="above" result="text"
        from toqito.states import orthogonal_singlet
        from toqito.state_props import is_ppt
        rho = orthogonal_singlet(2)
        print(rho.shape, is_ppt(rho, dim=[4, 4]))
        ```

    """
    d = shield_dim
    if d == 3:
        q_mats = []
        for k in range(3):
            phi = 2 * np.pi * (k + 1) / 3
            cos, sin = np.cos(phi), np.sin(phi)
            q_mats.append(np.array([[cos, sin, 0], [sin, -cos, 0], [0, 0, 1]]))
        e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
        s_vecs = [
            (np.kron(e_0, e_0) + np.kron(e_1, e_1)) / np.sqrt(2),
            (np.kron(e_0, e_1) - np.kron(e_1, e_0)) / np.sqrt(2),
            np.kron(e_2, e_2),
        ]
    elif d >= 2 and d & (d - 1) == 0:
        num_bits = d.bit_length() - 1
        pauli_x, eye_2 = np.array([[0, 1], [1, 0]]), np.identity(2)
        q_mats = []
        for k in range(d):
            q_mat = np.array([[1.0]])
            for bit in format(k, f"0{num_bits}b"):
                q_mat = np.kron(q_mat, pauli_x if bit == "1" else eye_2)
            q_mats.append(q_mat)
        s_vecs = [q_mat.reshape(-1, 1) / np.sqrt(d) for q_mat in q_mats]
    else:
        raise ValueError("The shield dimension must be 3 or a power of 2.")

    p_1 = np.sqrt(d) / (1 + np.sqrt(d))
    p_2 = 1 - p_1

    def ket(i: int, j: int, k: int, m: int) -> np.ndarray:
        return np.kron(np.kron(np.kron(basis(2, i), basis(2, j)), basis(d, k)), basis(d, m))

    rho = np.zeros((4 * d * d, 4 * d * d))
    for i in range(d):
        for j in range(d):
            z_vec = ket(0, 0, i, j) + sum(q_mats[j][i, k] * ket(1, 1, j, k) for k in range(d))
            rho += p_1 / (2 * d * d) * z_vec @ z_vec.T
    ket_01 = np.kron(basis(2, 0), basis(2, 1))
    for s_vec in s_vecs:
        vec = np.kron(ket_01, s_vec)
        rho += p_2 / (2 * d) * vec @ vec.T
    for i in range(d):
        vec = ket(1, 0, i, i)
        rho += p_2 / (2 * d) * vec @ vec.T

    return permute_systems(rho, [0, 2, 1, 3], dim=[2, 2, d, d])
