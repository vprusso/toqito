"""Compute the sandwiched conditional Rényi entropy of a bipartite quantum state."""

import numpy as np
from scipy.optimize import minimize

from toqito.matrix_ops.partial_trace import partial_trace
from toqito.matrix_props import is_density
from toqito.state_props._renyi_utils import (
    psd_matrix_power,
    support_is_subset,
    support_overlap,
    validate_bipartite_dim,
)
from toqito.state_props.von_neumann_entropy import von_neumann_entropy

_UPARROW_MIN_ALPHA = 0.5


def sandwiched_renyi_conditional_entropy(
    rho: np.ndarray,
    alpha: float,
    dim: int | list[int] | tuple | np.ndarray | None = None,
    variant: str = "downarrow",
    tol: float = 1e-9,
    max_iters: int = 500,
) -> float:
    r"""Compute the sandwiched conditional Rényi entropy of a bipartite state.

    For a bipartite density operator \(\rho_{AB}\), the downarrow sandwiched
    conditional Rényi entropy is defined by

    \[
        \widetilde{H}^{\downarrow}_{\alpha}(A|B)_{\rho}
        =
        -\widetilde{D}_{\alpha}\left(\rho_{AB}\parallel I_A \otimes \rho_B\right),
    \]

    while the uparrow variant is the optimization

    \[
        \widetilde{H}^{\uparrow}_{\alpha}(A|B)_{\rho}
        =
        \sup_{\sigma_B \succeq 0,\; \mathrm{Tr}(\sigma_B) = 1}
        -\widetilde{D}_{\alpha}\left(\rho_{AB}\parallel I_A \otimes \sigma_B\right),
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

    For `alpha=1`, both variants reduce to the conditional von Neumann entropy
    `H(AB) - H(B)`.

    The uparrow variant has no closed form, so it is computed by minimizing
    \(\widetilde{D}_{\alpha}(\rho_{AB}\|I_A \otimes \sigma_B)\) over density
    operators \(\sigma_B\) via `scipy.optimize.minimize` with the L-BFGS-B
    method. The density operator is parameterized as
    \(\sigma_B = A A^* / \mathrm{Tr}(A A^*)\) with a complex matrix \(A\). The
    sandwiched Rényi divergence is jointly convex in \((\rho, \sigma)\) for
    \(\alpha \in [1/2, \infty]\), so only that regime is supported for the
    uparrow variant; outside it, the optimization is non-convex and local
    minima need not be global. The edge cases `alpha=0` and `alpha=inf` are
    out of scope.

    Args:
        rho: Bipartite density operator.
        alpha: Rényi order. Positive orders are supported, and `alpha=1` is
            handled via the conditional von Neumann entropy. For the uparrow
            variant, the supported range is `alpha >= 1/2`.
        dim: Dimensions of the two subsystems. If `None`, both subsystems are
            assumed to have the same dimension.
        variant: Either `"downarrow"` or `"uparrow"`.
        tol: Gradient tolerance passed to the L-BFGS-B solver (uparrow only).
        max_iters: Maximum number of solver iterations (uparrow only).

    Returns:
        The sandwiched conditional Rényi entropy of `rho`.

    Raises:
        ValueError: If `rho` is not a density matrix, if `alpha <= 0`, if
            `variant` is not `"downarrow"` or `"uparrow"`, if `dim` does not
            describe a bipartite decomposition of `rho`, or if the uparrow
            variant is requested with `alpha < 1/2`.
        RuntimeError: If the uparrow optimizer fails to converge.

    Examples:
        Compute the downarrow variant for a Bell state:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_props import sandwiched_renyi_conditional_entropy

        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        print(sandwiched_renyi_conditional_entropy(rho, alpha=2, dim=2))
        ```

        Compute the uparrow variant for the same state:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_props import sandwiched_renyi_conditional_entropy

        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        print(sandwiched_renyi_conditional_entropy(rho, alpha=2, dim=2, variant="uparrow"))
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
    if variant not in {"downarrow", "uparrow"}:
        raise ValueError("`variant` must be either 'downarrow' or 'uparrow'.")

    dims = validate_bipartite_dim(rho, dim)
    rho_b = partial_trace(rho, [0], dims)

    if alpha == 1:
        return von_neumann_entropy(rho) - von_neumann_entropy(rho_b)

    if variant == "downarrow":
        return _sandwiched_renyi_conditional_entropy_downarrow(rho, rho_b, alpha, dims[0])

    if alpha < _UPARROW_MIN_ALPHA:
        raise ValueError(
            "The uparrow sandwiched conditional Rényi entropy is only supported for "
            "alpha >= 1/2, where the underlying optimization is convex."
        )
    return _sandwiched_renyi_conditional_entropy_uparrow(
        rho, rho_b, alpha, int(dims[0]), int(dims[1]), tol, max_iters
    )


def _sandwiched_renyi_conditional_entropy_downarrow(
    rho: np.ndarray, rho_b: np.ndarray, alpha: float, dim_a: int
) -> float:
    """Compute the downarrow sandwiched conditional Rényi entropy."""
    sigma = np.kron(np.eye(dim_a), rho_b)

    if alpha < 1 and support_overlap(rho, sigma) <= 0:
        return float("-inf")
    if alpha > 1 and not support_is_subset(rho, sigma):
        return float("-inf")

    sandwiched_power = (1 - alpha) / (2 * alpha)
    sigma_power = psd_matrix_power(sigma, sandwiched_power)
    sandwiched = sigma_power @ rho @ sigma_power
    trace_term = np.trace(psd_matrix_power(sandwiched, alpha))
    trace_term = float(np.real_if_close(trace_term))
    return -np.log2(trace_term) / (alpha - 1)


def _sandwiched_renyi_conditional_entropy_uparrow(
    rho: np.ndarray,
    rho_b: np.ndarray,
    alpha: float,
    dim_a: int,
    dim_b: int,
    tol: float,
    max_iters: int,
) -> float:
    """Compute the uparrow sandwiched conditional Rényi entropy via optimization."""
    identity_a = np.eye(dim_a)
    sandwich_exp = (1 - alpha) / (2 * alpha)
    penalty = 1e12
    n_real = dim_b * dim_b

    def params_to_sigma_b(params: np.ndarray) -> np.ndarray:
        a_real = params[:n_real].reshape(dim_b, dim_b)
        a_imag = params[n_real:].reshape(dim_b, dim_b)
        a_mat = a_real + 1j * a_imag
        mat = a_mat @ a_mat.conj().T
        trace_val = float(np.real_if_close(np.trace(mat)))
        if trace_val < 1e-15:
            return np.eye(dim_b) / dim_b
        return mat / trace_val

    def divergence(params: np.ndarray) -> float:
        sigma_b = params_to_sigma_b(params)
        sigma = np.kron(identity_a, sigma_b)

        if alpha < 1 and support_overlap(rho, sigma) <= 0:
            return penalty
        if alpha > 1 and not support_is_subset(rho, sigma):
            return penalty

        sigma_power = psd_matrix_power(sigma, sandwich_exp)
        sandwiched = sigma_power @ rho @ sigma_power
        trace_term = float(np.real(np.trace(psd_matrix_power(sandwiched, alpha))))
        if trace_term <= 0:
            return penalty
        return np.log2(trace_term) / (alpha - 1)

    eigvals, eigvecs = np.linalg.eigh((rho_b + rho_b.conj().T) / 2)
    eigvals = np.maximum(eigvals, 0.0)
    max_eig = float(np.max(eigvals)) if eigvals.size else 0.0
    if max_eig <= 0:
        eigvals = np.ones_like(eigvals) / dim_b
    else:
        eigvals = eigvals + 1e-3 * max_eig
        eigvals = eigvals / np.sum(eigvals)
    a_init = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
    x0 = np.concatenate([a_init.real.flatten(), a_init.imag.flatten()])

    result = minimize(
        divergence,
        x0,
        method="L-BFGS-B",
        options={"gtol": tol, "ftol": tol, "maxiter": max_iters},
    )

    result_fun = float(result.fun)
    if not np.isfinite(result_fun) or result_fun >= penalty / 2:
        raise RuntimeError(
            f"Uparrow sandwiched conditional Rényi entropy optimizer failed to converge: {result.message}"
        )

    return -result_fun
