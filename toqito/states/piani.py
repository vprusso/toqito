"""Benatti-Floreanini-Piani bound entangled state on two four-level systems."""

import numpy as np

from toqito.matrices import pauli
from toqito.states.max_entangled import max_entangled


def piani() -> np.ndarray:
    r"""Produce the Benatti-Floreanini-Piani state [@benatti2004nondecomposable].

    Write \(|\Psi_4^+\rangle = \frac{1}{2} \sum_{k=0}^3 |kk\rangle\) for the maximally entangled state of
    \(\mathbb{C}^4 \otimes \mathbb{C}^4\) and view the second factor as two qubits. For Pauli indices
    \(i, j \in \{0, 1, 2, 3\}\) with \(\sigma_0 = \mathbb{I}\), the generalized Bell projector is

    \[
        P_{ij} = |\Psi_{ij}\rangle\langle\Psi_{ij}|, \qquad
        |\Psi_{ij}\rangle = \left( \mathbb{I}_4 \otimes \sigma_i \otimes \sigma_j \right) |\Psi_4^+\rangle,
    \]

    and the state is the uniform mixture of six of them,

    \[
        \rho = \frac{1}{6} \left( P_{02} + P_{11} + P_{23} + P_{31} + P_{32} + P_{33} \right).
    \]

    It arises in the study of quantum dynamical semigroups that are not mixtures of decomposable ones
    [@benatti2004nondecomposable], and appears as a bound entangled example in [@imai2021bound]. The mixture
    is PPT with respect to the \(4 \otimes 4\) cut as it stands, has rank 6, and the realignment criterion
    certifies that it is entangled.

    Returns:
        The 16-by-16 state.

    Examples:
        The state is PPT and detected as entangled by the realignment criterion.

        ```python exec="1" source="above" result="text"
        from toqito.states import piani
        from toqito.state_props import is_ppt, is_separable
        rho = piani()
        print(is_ppt(rho, dim=[4, 4]), is_separable(rho, dim=[4, 4]))
        ```

    """
    max_ent = max_entangled(4)
    rho = np.zeros((16, 16), dtype=complex)
    for i, j in [(0, 2), (1, 1), (2, 3), (3, 1), (3, 2), (3, 3)]:
        vec = np.kron(np.identity(4), pauli([i, j])) @ max_ent
        rho += vec @ vec.conj().T
    return rho / 6
