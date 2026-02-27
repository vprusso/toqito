"""Sub-fidelity metric is a lower bound for the fidelity.

The sub-fidelity metric is a concave function and sub-multiplicative.
"""

import numpy as np

from toqito.matrix_props import is_density


def sub_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""Compute the sub fidelity of two density matrices [@Miszczak_2008_Sub].

    The sub-fidelity is a measure of similarity between density operators. It is defined as

    \[
        E(\rho, \sigma) = \text{Tr}(\rho \sigma) +
        \sqrt{2 \left[ \text{Tr}(\rho \sigma)^2 - \text{Tr}(\rho \sigma \rho \sigma) \right]},
    \]

    where \(\sigma\) and \(\rho\) are density matrices. The sub-fidelity serves as an lower bound for the
    fidelity.

    Examples:
        Consider the following pair of states:

        \[
            \rho = \frac{3}{4}|0\rangle \langle 0| +
                   \frac{1}{4}|1 \rangle \langle 1|
                    \quad \text{and} \quad
            \sigma = \frac{1}{8}|0 \rangle \langle 0| +
                     \frac{7}{8}|1 \rangle \langle 1|.
        \]

        Calculating the fidelity between the states \(\rho\) and \(\sigma\) as \(F(\rho, \sigma) \approx
        0.774\). This can be observed in `|toqito⟩` as

        ```python exec="1" source="above"
        from toqito.states import basis
        from toqito.state_metrics import fidelity

        e_0, e_1 = basis(2, 0), basis(2, 1)
        rho = 3 / 4 * e_0 @ e_0.conj().T + 1 / 4 * e_1 @ e_1.conj().T
        sigma = 1/8 * e_0 @ e_0.conj().T + 7/8 * e_1 @ e_1.conj().T

        print(fidelity(rho, sigma))
        ```

        As the sub-fidelity is a lower bound on the fidelity, that is \(E(\rho, \sigma) \leq F(\rho, \sigma)\), we can
        use `|toqito⟩` to observe that \(E(\rho, \sigma) \approx 0.599\leq F(\rho, \sigma \approx 0.774\).

        ```python exec="1" source="above"
        from toqito.states import basis
        from toqito.state_metrics import sub_fidelity

        e_0, e_1 = basis(2, 0), basis(2, 1)
        rho = 3 / 4 * e_0 @ e_0.conj().T + 1 / 4 * e_1 @ e_1.conj().T
        sigma = 1/8 * e_0 @ e_0.conj().T + 7/8 * e_1 @ e_1.conj().T

        print(sub_fidelity(rho, sigma))
        ```

    Raises:
        ValueError: If matrices are not of equal dimension.

    Args:
        rho: Density operator.
        sigma: Density operator.

    Returns:
        The sub-fidelity between `rho` and `sigma`.

    """
    # Perform some error checking.
    if not np.all(rho.shape == sigma.shape):
        raise ValueError("InvalidDim: `rho` and `sigma` must be matrices of the same size.")
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Sub-fidelity is only defined for density operators.")

    return np.real(
        np.trace(rho @ sigma) + np.sqrt(2 * (np.trace(rho @ sigma) ** 2 - np.trace(rho @ sigma @ rho @ sigma)))
    )
