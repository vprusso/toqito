"""Horodecki state generalized to two-by-d systems."""

import numpy as np

from toqito.states import basis


def generalized_horodecki(dim: int, b_param: float) -> np.ndarray:
    r"""Produce the Horodecki state generalized to \(\mathbb{C}^2 \otimes \mathbb{C}^d\) [@tura2012fourqubit].

    The \(2 \otimes 4\) Horodecki state [@horodecki1997separability] generalizes to
    \(\mathbb{C}^2 \otimes \mathbb{C}^d\) as follows. The entangled part mixes the \(d - 1\) vectors
    \(|\Psi_i\rangle = \frac{1}{\sqrt{2}} (|0, i\rangle + |1, i + 1\rangle)\), \(i = 0, \dots, d - 2\), with
    the corner state \(|1, 0\rangle\),

    \[
        \rho_{\text{insep}} = \frac{2}{2d - 1} \sum_{i=0}^{d-2} |\Psi_i\rangle\langle\Psi_i|
        + \frac{1}{2d - 1} |1, 0\rangle\langle 1, 0|,
    \]

    and is mixed with the product vector
    \(|\Phi_b\rangle = \frac{1}{\sqrt{2}} |0\rangle \otimes \big(\sqrt{1 - b} |0\rangle
    + \sqrt{1 + b} |d - 1\rangle\big)\) as

    \[
        \rho_{d, b} = \frac{(2d - 1) b \, \rho_{\text{insep}} + |\Phi_b\rangle\langle\Phi_b|}{(2d - 1) b + 1},
        \qquad 0 \leq b \leq 1.
    \]

    The state is PPT for every \(d \geq 2\) and \(b \in [0, 1]\). It is PPT entangled for \(d \geq 4\) and
    \(0 < b < 1\). It is separable for \(d = 2, 3\), where PPT implies separability, at \(b = 0\), where it
    is the product state \(|\Phi_0\rangle\), and at \(b = 1\), where it is the equal mixture of the \(d + 1\)
    product states \(\frac{1}{\sqrt{2d}} (e^{i\phi} |0\rangle + |1\rangle) \otimes \sum_{j} e^{ij\phi} |j\rangle\)
    with \(\phi = 2\pi k/(d+1)\). At \(d = 4\) it coincides with `toqito.states.horodecki` with `dim=[2, 4]`
    and the same parameter, up to reversing the order of the product basis, that is, conjugation by
    \(X \otimes J\) with \(X\) the qubit flip and \(J\) the anti-diagonal permutation of \(\mathbb{C}^4\).

    Args:
        dim: The dimension \(d \geq 2\) of the second system.
        b_param: The parameter \(b \in [0, 1]\).

    Returns:
        The generalized Horodecki state on \(\mathbb{C}^2 \otimes \mathbb{C}^d\).

    Raises:
        ValueError: If `dim` is smaller than 2 or `b_param` is outside \([0, 1]\).

    Examples:
        For \(d = 5\) and \(b = 0.5\) the state is PPT.

        ```python exec="1" source="above" result="text"
        from toqito.states import generalized_horodecki
        from toqito.state_props import is_ppt
        rho = generalized_horodecki(5, 0.5)
        print(rho.shape, is_ppt(rho, dim=[2, 5]))
        ```

    """
    if dim < 2:
        raise ValueError("The dimension of the second system must be at least 2.")
    if not 0 <= b_param <= 1:
        raise ValueError("The parameter b must lie in [0, 1].")

    rho_insep = np.zeros((2 * dim, 2 * dim))
    for i in range(dim - 1):
        psi = (np.kron(basis(2, 0), basis(dim, i)) + np.kron(basis(2, 1), basis(dim, i + 1))) / np.sqrt(2)
        rho_insep += 2 / (2 * dim - 1) * psi @ psi.T
    corner = np.kron(basis(2, 1), basis(dim, 0))
    rho_insep += corner @ corner.T / (2 * dim - 1)

    phi = np.kron(basis(2, 0), np.sqrt(1 - b_param) * basis(dim, 0) + np.sqrt(1 + b_param) * basis(dim, dim - 1))
    phi /= np.sqrt(2)
    return ((2 * dim - 1) * b_param * rho_insep + phi @ phi.T) / ((2 * dim - 1) * b_param + 1)
