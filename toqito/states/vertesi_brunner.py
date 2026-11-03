"""Vertesi-Brunner bound entangled state that violates a Bell inequality."""

import numpy as np

from toqito.states.basis import basis


def vertesi_brunner() -> np.ndarray:
    r"""Produce the Vertesi-Brunner state [@vertesi2014disproving].

    Peres conjectured that no bound entangled state violates a Bell inequality. This two-qutrit state, which
    is PPT and hence bound entangled, violates one, disproving the conjecture. Its spectral decomposition is
    \(\rho = \sum_{i=1}^4 \lambda_i |\psi_i\rangle\langle\psi_i|\) with \(a = \sqrt{131/2}\) and

    \[
        \begin{aligned}
            |\psi_1\rangle &= \frac{1}{\sqrt{2}} \left( |00\rangle + |11\rangle \right), \\
            |\psi_2\rangle &= \frac{a}{12} \left( |01\rangle + |10\rangle \right) + \frac{1}{60} |02\rangle
                - \frac{3}{10} |21\rangle, \\
            |\psi_3\rangle &= \frac{a}{12} \left( |00\rangle - |11\rangle \right) + \frac{1}{60} |12\rangle
                + \frac{3}{10} |20\rangle, \\
            |\psi_4\rangle &= \frac{1}{\sqrt{3}} \left( -|01\rangle + |10\rangle + |22\rangle \right),
        \end{aligned}
    \]

    with eigenvalues \(\lambda = (3257/6884, \ 450/1721, \ 450/1721, \ 27/6884)\). The state has rank 4 and
    the same spectrum as `toqito.states.moroder_steering` at \(m_1 = 1/60\), \(m_2 = 3/10\), although the two
    matrices differ.

    Returns:
        The 9-by-9 Vertesi-Brunner state.

    Examples:
        The state is PPT and detected as entangled by the realignment criterion.

        ```python exec="1" source="above" result="text"
        from toqito.states import vertesi_brunner
        from toqito.state_props import is_ppt, is_separable
        rho = vertesi_brunner()
        print(is_ppt(rho, dim=[3, 3]), is_separable(rho, dim=[3, 3]))
        ```

    """

    def ket(i: int, j: int) -> np.ndarray:
        return np.kron(basis(3, i), basis(3, j))

    a_param = np.sqrt(131 / 2)
    vecs = [
        (ket(0, 0) + ket(1, 1)) / np.sqrt(2),
        a_param / 12 * (ket(0, 1) + ket(1, 0)) + ket(0, 2) / 60 - 3 / 10 * ket(2, 1),
        a_param / 12 * (ket(0, 0) - ket(1, 1)) + ket(1, 2) / 60 + 3 / 10 * ket(2, 0),
        (-ket(0, 1) + ket(1, 0) + ket(2, 2)) / np.sqrt(3),
    ]
    eigenvalues = [3257 / 6884, 450 / 1721, 450 / 1721, 27 / 6884]
    return sum(lam * vec @ vec.T for lam, vec in zip(eigenvalues, vecs))
