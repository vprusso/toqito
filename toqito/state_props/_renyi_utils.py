"""Internal helper functions for Rényi entropy calculations."""

import numpy as np


def validate_bipartite_dim(
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


def psd_matrix_power(mat: np.ndarray, power: float, tol: float = 1e-12) -> np.ndarray:
    """Apply a real power to a PSD matrix on its support."""
    eigvals, eigvecs = np.linalg.eigh((mat + mat.conj().T) / 2)
    powered = np.zeros_like(eigvals, dtype=float)
    positive = eigvals > tol
    powered[positive] = eigvals[positive] ** power
    return eigvecs @ np.diag(powered) @ eigvecs.conj().T


def support_overlap(mat_1: np.ndarray, mat_2: np.ndarray, tol: float = 1e-12) -> float:
    """Return the overlap between the supports of two PSD matrices."""
    proj_1 = support_projector(mat_1, tol)
    proj_2 = support_projector(mat_2, tol)
    return float(np.real_if_close(np.trace(proj_1 @ proj_2)))


def support_is_subset(mat_1: np.ndarray, mat_2: np.ndarray, tol: float = 1e-12) -> bool:
    """Check whether support(mat_1) is contained in support(mat_2)."""
    proj_1 = support_projector(mat_1, tol)
    proj_2 = support_projector(mat_2, tol)
    leak = np.trace(proj_1 @ (np.eye(mat_1.shape[0]) - proj_2))
    return float(np.real_if_close(leak)) <= tol


def support_projector(mat: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return the orthogonal projector onto the support of a PSD matrix."""
    eigvals, eigvecs = np.linalg.eigh((mat + mat.conj().T) / 2)
    positive = eigvals > tol
    if not np.any(positive):
        return np.zeros_like(mat, dtype=complex)
    return eigvecs[:, positive] @ eigvecs[:, positive].conj().T
