"""Compute the sandwiched conditional Rényi entropy of a bipartite quantum state."""

import numpy as np

from toqito.matrix_ops.partial_trace import partial_trace
from toqito.matrix_props import is_density
from toqito.state_props.petz_renyi_conditional_entropy import (
    _psd_matrix_power,
    _support_is_subset,
    _support_overlap,
    _validate_bipartite_dim,
)
from toqito.state_props.von_neumann_entropy import von_neumann_entropy


def sandwiched_renyi_conditional_entropy(
    rho: np.ndarray,
    alpha: float,
    dim: int | list[int] | tuple | np.ndarray | None = None,
    variant: str = "downarrow",
) -> float:
    r"""Compute the sandwiched conditional Rényi entropy of a bipartite state.

    For a bipartite density operator \(\rho_{AB}\), the downarrow sandwiched
    conditional Rényi entropy is defined by

    \[
        \widetilde{H}^{\downarrow}_{\alpha}(A|B)_{\rho}
        =
        -\widetilde{D}_{\alpha}\left(\rho_{AB}\parallel I_A \otimes \rho_B\right),
    \]

    where \(\widetilde{D}_{\alpha}\) is the sandwiched Rényi divergence

    \[
        \widetilde{D}_{\alpha}(\rho\|\sigma)
        =
        \frac{1}{\alpha - 1}
        \log_2\left(
            \mathrm{Tr}\left[
                \left(
                    \sigma^{(1-\alpha)/(2\alpha)}
                    \rho
                    \sigma^{(1-\alpha)/(2\alpha)}
                \right)^\alpha
            \right]
        \right).
    \]

    This function currently implements only the downarrow variant. The uparrow
    variant requires an optimization over density operators on subsystem `B` and
    is intentionally rejected until that optimization is implemented.

    For `alpha=1`, the downarrow variant recovers the conditional von Neumann
    entropy `H(AB) - H(B)`.

    Args:
        rho: Bipartite density operator.
        alpha: Rényi order. This function supports positive orders and handles
            `alpha=1` via the conditional von Neumann entropy.
        dim: Dimensions of the two subsystems. If `None`, both subsystems are
            assumed to have the same dimension.
        variant: Currently only `"downarrow"` is supported.

    Returns:
        The sandwiched conditional Rényi entropy of `rho`.

    Raises:
        ValueError: If `rho` is not a density matrix, if `alpha <= 0`, if
            `variant` is not `"downarrow"`, or if `dim` does not describe a
            bipartite decomposition of `rho`.

    Examples:
        Compute the downarrow variant for a Bell state:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_props import sandwiched_renyi_conditional_entropy

        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        print(sandwiched_renyi_conditional_entropy(rho, alpha=2, dim=2))
        ```

        For a product state, the conditional entropy matches the Rényi entropy
        of the first subsystem:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_props import sandwiched_renyi_conditional_entropy

        rho = np.diag([1 / 2, 1 / 2, 0, 0])
        print(sandwiched_renyi_conditional_entropy(rho, alpha=2, dim=2))
        ```

    """
    if not is_density(rho):
        raise ValueError("Sandwiched conditional Rényi entropy is only defined for density operators.")
    if alpha <= 0:
        raise ValueError("Sandwiched conditional Rényi entropy is only defined for positive orders.")
    if variant != "downarrow":
        raise ValueError("Only the 'downarrow' sandwiched conditional Rényi entropy is currently supported.")

    dims = _validate_bipartite_dim(rho, dim)
    rho_b = partial_trace(rho, [0], dims)

    if alpha == 1:
        return von_neumann_entropy(rho) - von_neumann_entropy(rho_b)

    return _sandwiched_renyi_conditional_entropy_downarrow(rho, rho_b, alpha, dims[0])


def _sandwiched_renyi_conditional_entropy_downarrow(
    rho: np.ndarray, rho_b: np.ndarray, alpha: float, dim_a: int
) -> float:
    """Compute the downarrow sandwiched conditional Rényi entropy."""
    sigma = np.kron(np.eye(dim_a), rho_b)

    if alpha < 1 and _support_overlap(rho, sigma) <= 0:
        return float("-inf")
    if alpha > 1 and not _support_is_subset(rho, sigma):
        return float("-inf")

    sandwiched_power = (1 - alpha) / (2 * alpha)
    sigma_power = _psd_matrix_power(sigma, sandwiched_power)
    sandwiched = sigma_power @ rho @ sigma_power
    trace_term = np.trace(_psd_matrix_power(sandwiched, alpha))
    trace_term = float(np.real_if_close(trace_term))
    return -np.log2(trace_term) / (alpha - 1)
