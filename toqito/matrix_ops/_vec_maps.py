"""Sparse linear maps acting on the row-major vectorization of a matrix.

Applying a partial trace or partial transpose to a cvxpy expression as a sparse matrix on `vec(X)` canonicalizes
far faster than assembling the result from scalar entries or from cvxpy's own block-indexing atoms.
"""

import cvxpy
import numpy as np
from scipy import sparse


def _vec_partial_trace(dims: list[int], sys: list[int]) -> sparse.csr_array:
    r"""Return the sparse matrix mapping the row-major vectorization of \(X\) to that of \(\text{Tr}_{sys}(X)\)."""
    num = len(dims)
    keep = [i for i in range(num) if i not in sys]
    dim_total = int(np.prod(dims))
    dim_keep = int(np.prod([dims[i] for i in keep]))
    dim_trace = dim_total // dim_keep
    idx = np.arange(dim_total**2).reshape(dims + dims)
    idx = idx.transpose(keep + [num + i for i in keep] + sys + [num + i for i in sys])
    cols = np.diagonal(idx.reshape(dim_keep**2, dim_trace, dim_trace), axis1=1, axis2=2)
    rows = np.repeat(np.arange(dim_keep**2), dim_trace)
    return sparse.csr_array((np.ones(rows.size), (rows, cols.ravel())), shape=(dim_keep**2, dim_total**2))


def _vec_partial_transpose_perm(dims: list[int], sys: int | list[int]) -> np.ndarray:
    r"""Return `perm` such that the row-major vectorization of \(T_{sys}(X)\) is `vec(X)[perm]`.

    Transposes of distinct subsystems commute, so a list of subsystems is handled by transposing each in turn.
    """
    num = len(dims)
    dim_total = int(np.prod(dims))
    idx = np.arange(dim_total**2).reshape(dims + dims)
    for system in [sys] if isinstance(sys, int) else sys:
        idx = np.swapaxes(idx, system, num + system)
    return idx.ravel()


def _vec_kron_identity(dim: int, dim_identity: int) -> sparse.csr_array:
    r"""Return the sparse matrix mapping the row-major vectorization of \(A\) to that of \(A \otimes \mathbb{I}\)."""
    rows_a, cols_a = np.divmod(np.arange(dim**2), dim)
    diag = np.arange(dim_identity)
    rows = (
        (rows_a[:, None] * dim_identity + diag) * (dim * dim_identity) + cols_a[:, None] * dim_identity + diag
    ).ravel()
    cols = np.repeat(np.arange(dim**2), dim_identity)
    return sparse.csr_array((np.ones(rows.size), (rows, cols)), shape=((dim * dim_identity) ** 2, dim**2))


def _apply_vec_map(vec_map: sparse.csr_array, var: cvxpy.Expression, out_dim: int) -> cvxpy.Expression:
    """Apply a linear map, given as a matrix acting on row-major vectorizations, to a square CVXPY expression."""
    return cvxpy.reshape(vec_map @ cvxpy.vec(var, order="C"), (out_dim, out_dim), order="C")
