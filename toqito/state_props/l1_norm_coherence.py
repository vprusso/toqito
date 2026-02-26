"""Computes the l1-norm of coherence of a quantum state."""

import numpy as np

from toqito.matrix_ops import to_density_matrix


def l1_norm_coherence(rho: np.ndarray) -> float:
    r"""Compute the l1-norm of coherence of a quantum state [@Rana_2017_Log].

    The \(\ell_1\)-norm of coherence of a quantum state \(\rho\) is
    defined as

    \[
        C_{\ell_1}(\rho) = \sum_{i \not= j} \left|\rho_{i,j}\right|,
    \]

    where \(\rho_{i,j}\) is the \((i,j)^{th}\)-entry of \(\rho\)
    in the standard basis.

    The \(\ell_1\)-norm of coherence is the sum of the absolute values of
    the sum of the absolute values of the off-diagonal entries of the density
    matrix `rho` in the standard basis.

    This function was adapted from QETLAB.

    Examples:
    The largest possible value of the \(\ell_1\)-norm of coherence on
    \(d\)-dimensional states is \(d-1\), and is attained exactly by
    the "maximally coherent states": pure states whose entries all have the
    same absolute value.

    ```python exec="1" source="above"
    from toqito.state_props import l1_norm_coherence
    import numpy as np
    # Maximally coherent state.
    v = np.ones((3,1))/np.sqrt(3)
    print(l1_norm_coherence(v))
    ```

    Args:
        rho: A matrix or vector.

    Returns:
        The l1-norm coherence of `rho`.

    """
    rho = to_density_matrix(rho)
    return np.sum(np.sum(np.abs(rho))) - np.trace(rho)
