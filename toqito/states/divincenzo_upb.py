"""Six-parameter family of unextendible product bases on two qutrits."""

import numpy as np


def divincenzo_upb(
    gamma_a: float, theta_a: float, phi_a: float, gamma_b: float, theta_b: float, phi_b: float
) -> list[np.ndarray]:
    r"""Produce a member of the six-parameter family of two-qutrit UPBs [@divincenzo2003unextendible].

    Every unextendible product basis (UPB) on \(\mathbb{C}^3 \otimes \mathbb{C}^3\) consists of five product vectors
    \(|\psi_i\rangle = |\alpha_i\rangle \otimes |\beta_i\rangle\) with the orthogonality structure of the
    Tile and Pyramid bases, and [@divincenzo2003unextendible] shows that, up to local unitaries, they are
    all of the form

    \[
        \begin{aligned}
            |\alpha_0\rangle &= |0\rangle, &
            |\beta_0\rangle &= |1\rangle, \\
            |\alpha_1\rangle &= |1\rangle, &
            |\beta_1\rangle &= \sin\gamma_B \sin\theta_B |0\rangle + \cos\gamma_B e^{i\varphi_B} |1\rangle
                - \sin\gamma_B \cos\theta_B |2\rangle, \\
            |\alpha_2\rangle &= \cos\theta_A |0\rangle + \sin\theta_A |2\rangle, &
            |\beta_2\rangle &= |0\rangle, \\
            |\alpha_3\rangle &= \sin\gamma_A \sin\theta_A |0\rangle + \cos\gamma_A e^{i\varphi_A} |1\rangle
                - \sin\gamma_A \cos\theta_A |2\rangle, &
            |\beta_3\rangle &= \cos\theta_B |0\rangle + \sin\theta_B |2\rangle, \\
            |\alpha_4\rangle &= \frac{1}{N_A}\left(\sin\gamma_A \cos\theta_A e^{i\varphi_A} |1\rangle
                + \cos\gamma_A |2\rangle\right), &
            |\beta_4\rangle &= \frac{1}{N_B}\left(\sin\gamma_B \cos\theta_B e^{i\varphi_B} |1\rangle
                + \cos\gamma_B |2\rangle\right),
        \end{aligned}
    \]

    where \(N_{A} = \sqrt{\cos^2\gamma_A + \sin^2\gamma_A \cos^2\theta_A}\) and likewise for \(N_B\).
    The five vectors are orthonormal for every choice of the angles, and they form a UPB exactly when
    \(\cos\) and \(\sin\) of each of \(\gamma_A, \theta_A, \gamma_B, \theta_B\) are all non-zero.

    With \(\varphi_A = \varphi_B = 0\) and \(\gamma_A = \theta_A = \gamma_B = \theta_B = 3\pi/4\) the
    member is equivalent to the Tile basis, and with
    \(\gamma_A = \theta_A = \gamma_B = \theta_B = \arccos\big((\sqrt{5} - 1)/2\big)\) to the Pyramid basis,
    in both cases up to local unitaries and an exchange of the two parties: the moduli of the inner products
    of the local factors agree after relabeling the five vectors.

    Args:
        gamma_a: Angle \(\gamma_A\) of the first party.
        theta_a: Angle \(\theta_A\) of the first party.
        phi_a: Phase \(\varphi_A\) of the first party.
        gamma_b: Angle \(\gamma_B\) of the second party.
        theta_b: Angle \(\theta_B\) of the second party.
        phi_b: Phase \(\varphi_B\) of the second party.

    Returns:
        A list of the five product vectors as column vectors.

    Raises:
        ValueError: If \(\cos\) or \(\sin\) of one of the four angles \(\gamma_A, \theta_A, \gamma_B,
            \theta_B\) vanishes, in which case the vectors do not form a UPB.

    Examples:
        A generic member is a UPB.

        ```python exec="1" source="above" result="text"
        from toqito.states import divincenzo_upb
        from toqito.state_props import is_unextendible_product_basis
        vecs = divincenzo_upb(0.4, 1.1, 0.3, 0.9, 0.7, 2.0)
        print(is_unextendible_product_basis(vecs, [3, 3])[0])
        ```

    """
    for angle in (gamma_a, theta_a, gamma_b, theta_b):
        if np.isclose(np.cos(angle), 0) or np.isclose(np.sin(angle), 0):
            raise ValueError("cos and sin of gamma_a, theta_a, gamma_b and theta_b must all be non-zero.")

    def party(gamma: float, theta: float, phi: float) -> list[np.ndarray]:
        phase = np.exp(1j * phi)
        norm = np.sqrt(np.cos(gamma) ** 2 + np.sin(gamma) ** 2 * np.cos(theta) ** 2)
        return [
            np.array([[1], [0], [0]], dtype=complex),
            np.array([[0], [1], [0]], dtype=complex),
            np.array([[np.cos(theta)], [0], [np.sin(theta)]], dtype=complex),
            np.array([[np.sin(gamma) * np.sin(theta)], [np.cos(gamma) * phase], [-np.sin(gamma) * np.cos(theta)]]),
            np.array([[0], [np.sin(gamma) * np.cos(theta) * phase / norm], [np.cos(gamma) / norm]]),
        ]

    alpha = party(gamma_a, theta_a, phi_a)
    beta = party(gamma_b, theta_b, phi_b)
    # The second party uses the same five vectors with the roles of the first two and the next two swapped.
    order_b = [1, 3, 0, 2, 4]
    return [np.kron(alpha[i], beta[order_b[i]]) for i in range(5)]
