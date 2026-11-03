"""Steerable bound entangled states of Moroder, Gittsovich, Huber and Guehne."""

import numpy as np

from toqito.states.basis import basis


def moroder_steering(m1: float, m2: float) -> np.ndarray:
    r"""Produce a member of the two-parameter family of steerable PPT states [@moroder2014steering].

    The stronger Peres conjecture held that bound entangled states cannot be used to demonstrate EPR
    steering. This two-qutrit family is PPT by construction and contains steerable members, which
    disproves it. With parameters \(m_1, m_2 \geq 0\) and \(m_3 = \sqrt{(1 - m_1^2 - m_2^2)/2}\), let

    \[
        \begin{aligned}
            |\psi_1\rangle &= \frac{1}{\sqrt{2}} \left( |12\rangle + |21\rangle \right), \\
            |\psi_2\rangle &= \frac{1}{\sqrt{3}} \left( |00\rangle + |11\rangle - |22\rangle \right), \\
            |\psi_3\rangle &= m_1 |01\rangle + m_2 |10\rangle + m_3 \left( |11\rangle + |22\rangle \right), \\
            |\tilde\psi_3\rangle &= m_1 |02\rangle - m_2 |20\rangle + m_3 \left( |21\rangle - |12\rangle \right),
        \end{aligned}
    \]

    and

    \[
        \rho = \lambda_1 |\psi_1\rangle\langle\psi_1| + \lambda_2 |\psi_2\rangle\langle\psi_2|
            + \lambda_3 \left( |\psi_3\rangle\langle\psi_3| + |\tilde\psi_3\rangle\langle\tilde\psi_3| \right).
    \]

    Invariance under partial transposition fixes the weights: with \(D = 4 - 2m_1^2 + m_1 m_2 - 2m_2^2\),

    \[
        \lambda_1 = 1 - \frac{2 + 3 m_1 m_2}{D}, \qquad \lambda_3 = \frac{1}{D},
        \qquad \lambda_2 = 1 - \lambda_1 - 2\lambda_3.
    \]

    The weights are non-negative exactly when \(m_1^2 + m_2^2 + m_1 m_2 \leq 1\), which is the admissible
    region. The state has rank 4 except where a weight vanishes, as at \(m_1 = m_2 = 0\) where
    \(\lambda_2 = 0\). At \(m_1 = 1/60\) and \(m_2 = 3/10\) the state has the spectrum of
    `toqito.states.vertesi_brunner`.

    Args:
        m1: The parameter \(m_1 \geq 0\).
        m2: The parameter \(m_2 \geq 0\).

    Returns:
        The 9-by-9 state.

    Raises:
        ValueError: If \(m_1 < 0\), \(m_2 < 0\), or \(m_1^2 + m_2^2 + m_1 m_2 > 1\).

    Examples:
        The state is PPT and detected as entangled by the realignment criterion.

        ```python exec="1" source="above" result="text"
        from toqito.states import moroder_steering
        from toqito.state_props import is_ppt, is_separable
        rho = moroder_steering(0.3, 0.4)
        print(is_ppt(rho, dim=[3, 3]), is_separable(rho, dim=[3, 3]))
        ```

    """
    if m1 < 0 or m2 < 0 or m1**2 + m2**2 + m1 * m2 > 1:
        raise ValueError("The parameters must satisfy m1, m2 >= 0 and m1**2 + m2**2 + m1 * m2 <= 1.")

    def ket(i: int, j: int) -> np.ndarray:
        return np.kron(basis(3, i), basis(3, j))

    m3 = np.sqrt((1 - m1**2 - m2**2) / 2)
    psi_1 = (ket(1, 2) + ket(2, 1)) / np.sqrt(2)
    psi_2 = (ket(0, 0) + ket(1, 1) - ket(2, 2)) / np.sqrt(3)
    psi_3 = m1 * ket(0, 1) + m2 * ket(1, 0) + m3 * (ket(1, 1) + ket(2, 2))
    psi_3_tilde = m1 * ket(0, 2) - m2 * ket(2, 0) + m3 * (ket(2, 1) - ket(1, 2))

    denom = 4 - 2 * m1**2 + m1 * m2 - 2 * m2**2
    lam_1 = 1 - (2 + 3 * m1 * m2) / denom
    lam_3 = 1 / denom
    lam_2 = 1 - lam_1 - 2 * lam_3
    return lam_1 * psi_1 @ psi_1.T + lam_2 * psi_2 @ psi_2.T + lam_3 * (psi_3 @ psi_3.T + psi_3_tilde @ psi_3_tilde.T)
