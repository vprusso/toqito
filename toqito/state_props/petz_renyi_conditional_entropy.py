"""Compute the Petz conditional Rényi entropy of a bipartite quantum state."""

import numpy as np

from toqito.matrix_ops.partial_trace import partial_trace
from toqito.matrix_props import is_density
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

    """
    if not is_density(rho):
        raise ValueError("Petz conditional Rényi entropy is only defined for density operators.")
    if alpha <= 0:
        raise ValueError("Petz conditional Rényi entropy is only defined for positive orders.")
    if variant not in {"downarrow", "uparrow"}:
        raise ValueError("`variant` must be either 'downarrow' or 'uparrow'.")

    dims = _validate_bipartite_dim(rho, dim)
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

    if alpha < 1 and _support_overlap(rho, sigma) <= 0:
        return float("-inf")
    if alpha > 1 and not _support_is_subset(rho, sigma):
        return float("-inf")

    trace_term = np.trace(_psd_matrix_power(rho, alpha) @ _psd_matrix_power(sigma, 1 - alpha))
    trace_term = float(np.real_if_close(trace_term))
    return -np.log2(trace_term) / (alpha - 1)


def _petz_renyi_conditional_entropy_uparrow(rho: np.ndarray, alpha: float, dims: np.ndarray) -> float:
    """Compute the uparrow Petz conditional Rényi entropy via the closed-form formula."""
    rho_alpha = _psd_matrix_power(rho, alpha)
    traced = partial_trace(rho_alpha, [0], dims)
    traced_power = _psd_matrix_power(traced, 1 / alpha)
    trace_term = float(np.real_if_close(np.trace(traced_power)))
    return alpha * np.log2(trace_term) / (1 - alpha)


def _validate_bipartite_dim(
    rho: np.ndarray, dim: int | list[int] | tuple | np.ndarray | None
) -> np.ndarray:
    """Validate and normalize a bipartite dimension specification."""
    n = rho.shape[0]

    if dim is None:
        d = int(round(np.sqrt(n)))
        if d * d != n:
            raise ValueError("Cannot infer bipartite subsystem dimensions directly. Please provide `dim`.")
        return np.array([d, d], dtype=int)

    if isinstance(dim, int):
        if dim <= 0 or n % dim != 0:
            raise ValueError("If `dim` is a scalar, it must be a positive divisor of the matrix dimension.")
        return np.array([dim, n // dim], dtype=int)

    dims = np.asarray(dim)
    if dims.ndim != 1 or len(dims) != 2:
        raise ValueError("`dim` must describe exactly two subsystem dimensions.")
    if not np.issubdtype(dims.dtype, np.integer):
        raise ValueError("`dim` must contain integer subsystem dimensions.")
    if np.any(dims <= 0):
        raise ValueError("Subsystem dimensions in `dim` must be positive.")
    if int(np.prod(dims)) != n:
        raise ValueError("The product of `dim` must match the dimension of `rho`.")
    return dims.astype(int)


def _psd_matrix_power(mat: np.ndarray, power: float, tol: float = 1e-12) -> np.ndarray:
    """Apply a real power to a PSD matrix on its support."""
    eigvals, eigvecs = np.linalg.eigh((mat + mat.conj().T) / 2)
    powered = np.zeros_like(eigvals, dtype=float)
    positive = eigvals > tol
    powered[positive] = eigvals[positive] ** power
    return eigvecs @ np.diag(powered) @ eigvecs.conj().T


def _support_overlap(mat_1: np.ndarray, mat_2: np.ndarray, tol: float = 1e-12) -> float:
    """Return the overlap between the supports of two PSD matrices."""
    proj_1 = _support_projector(mat_1, tol)
    proj_2 = _support_projector(mat_2, tol)
    return float(np.real_if_close(np.trace(proj_1 @ proj_2)))


def _support_is_subset(mat_1: np.ndarray, mat_2: np.ndarray, tol: float = 1e-12) -> bool:
    """Check whether support(mat_1) is contained in support(mat_2)."""
    proj_1 = _support_projector(mat_1, tol)
    proj_2 = _support_projector(mat_2, tol)
    leak = np.trace(proj_1 @ (np.eye(mat_1.shape[0]) - proj_2))
    return float(np.real_if_close(leak)) <= tol


def _support_projector(mat: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return the orthogonal projector onto the support of a PSD matrix."""
    eigvals, eigvecs = np.linalg.eigh((mat + mat.conj().T) / 2)
    positive = eigvals > tol
    if not np.any(positive):
        return np.zeros_like(mat, dtype=complex)
    return eigvecs[:, positive] @ eigvecs[:, positive].conj().T
