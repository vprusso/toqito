"""Compute the Petz conditional Rényi entropy of a bipartite quantum state."""

import numpy as np

from toqito.matrix_ops.partial_trace import partial_trace
from toqito.matrix_props import is_density
from toqito.state_props._renyi_utils import (
    psd_matrix_power,
    support_is_subset,
    support_overlap,
    validate_bipartite_dim,
)
from toqito.state_props.von_neumann_entropy import von_neumann_entropy


def petz_renyi_conditional_entropy(
    rho: np.ndarray,
    alpha: float,
    dim: int | list[int] | tuple | np.ndarray | None = None,
    variant: str = "downarrow",
) -> float:
    r"""Compute the Petz conditional Rényi entropy of a bipartite state.

    For a bipartite density operator \(\rho_{AB}\), the downarrow Petz conditional Rényi
    entropy is defined by

    \[
        \overline{H}^{\downarrow}_{\alpha}(A|B)_{\rho}
        =
        -\overline{D}_{\alpha}\left(\rho_{AB}\parallel I_A \otimes \rho_B\right),
    \]

    while the uparrow version is

    \[
        \overline{H}^{\uparrow}_{\alpha}(A|B)_{\rho}
        =
        \sup_{\sigma_B \succeq 0,\; \mathrm{Tr}(\sigma_B)=1}
        -\overline{D}_{\alpha}\left(\rho_{AB}\parallel I_A \otimes \sigma_B\right).
    \]

    The uparrow case admits the closed-form expression

    \[
        \overline{H}^{\uparrow}_{\alpha}(A|B)_{\rho}
        =
        \frac{\alpha}{1-\alpha}
        \log_2\left(
            \mathrm{Tr}\left[\left(\mathrm{Tr}_A\left[\rho_{AB}^{\alpha}\right]\right)^{1/\alpha}\right]
        \right).
    \]

    For `alpha=1`, both variants recover the conditional von Neumann entropy
    `H(AB) - H(B)`.

    Args:
        rho: Bipartite density operator.
        alpha: Rényi order. This function supports positive orders and handles `alpha=1`
            via the conditional von Neumann entropy.
        dim: Dimensions of the two subsystems. If `None`, both subsystems are assumed to
            have the same dimension.
        variant: Either `"downarrow"` or `"uparrow"`.

    Returns:
        The Petz conditional Rényi entropy of `rho`.

    Raises:
        ValueError: If `rho` is not a density matrix, if `alpha <= 0`, if `variant` is
            invalid, or if `dim` does not describe a bipartite decomposition of `rho`.

    Examples:
        Compute the downarrow variant for a Bell state using a scalar subsystem
        dimension:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_props import petz_renyi_conditional_entropy

        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        print(petz_renyi_conditional_entropy(rho, alpha=2, dim=2, variant="downarrow"))
        ```

        Compute the uparrow variant for the same state using a 2-element list for
        `dim`:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_props import petz_renyi_conditional_entropy

        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        print(petz_renyi_conditional_entropy(rho, alpha=2, dim=[2, 2], variant="uparrow"))
        ```

        For a product state, the conditional entropy matches the entropy of the
        first subsystem:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_props import petz_renyi_conditional_entropy

        rho = np.diag([1 / 2, 1 / 2, 0, 0])
        print(petz_renyi_conditional_entropy(rho, alpha=2, dim=2, variant="downarrow"))
        ```

    """
    if not is_density(rho):
        raise ValueError("Petz conditional Rényi entropy is only defined for density operators.")
    if alpha <= 0:
        raise ValueError("Petz conditional Rényi entropy is only defined for positive orders.")
    if variant not in {"downarrow", "uparrow"}:
        raise ValueError("`variant` must be either 'downarrow' or 'uparrow'.")

    dims = validate_bipartite_dim(rho, dim)
    rho_b = partial_trace(rho, [0], dims)

    if alpha == 1:
        return von_neumann_entropy(rho) - von_neumann_entropy(rho_b)
    if variant == "uparrow":
        return _petz_renyi_conditional_entropy_uparrow(rho, alpha, dims)
    return _petz_renyi_conditional_entropy_downarrow(rho, rho_b, alpha, dims[0])


def _petz_renyi_conditional_entropy_downarrow(
    rho: np.ndarray, rho_b: np.ndarray, alpha: float, dim_a: int
) -> float:
    """Compute the downarrow Petz conditional Rényi entropy."""
    sigma = np.kron(np.eye(dim_a), rho_b)

    if alpha < 1 and support_overlap(rho, sigma) <= 0:
        return float("-inf")
    if alpha > 1 and not support_is_subset(rho, sigma):
        return float("-inf")

    trace_term = np.trace(psd_matrix_power(rho, alpha) @ psd_matrix_power(sigma, 1 - alpha))
    trace_term = float(np.real_if_close(trace_term))
    return -np.log2(trace_term) / (alpha - 1)


def _petz_renyi_conditional_entropy_uparrow(rho: np.ndarray, alpha: float, dims: np.ndarray) -> float:
    """Compute the uparrow Petz conditional Rényi entropy via the closed-form formula."""
    rho_alpha = psd_matrix_power(rho, alpha)
    traced = partial_trace(rho_alpha, [0], dims)
    traced_power = psd_matrix_power(traced, 1 / alpha)
    trace_term = float(np.real_if_close(np.trace(traced_power)))
    return alpha * np.log2(trace_term) / (1 - alpha)
