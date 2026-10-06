"""Shared internal helpers for symmetric-extension SDPs parameterized on the symmetric subspace."""

from math import comb

import cvxpy
import numpy as np
from scipy import sparse

from toqito.matrix_ops import partial_transpose


def _symmetric_isometry(dim: int, level: int) -> sparse.csr_array:
    r"""Return a sparse isometry onto the symmetric subspace of \((\mathbb{C}^{d})^{\otimes k}\).

    Column \(j\) is the normalized uniform superposition of all computational basis states
    whose digits are a permutation of the \(j\)-th multiset, so \(V^T V = \mathbb{I}\) and
    \(V V^T\) is the projector onto the symmetric subspace. Every basis state belongs to exactly
    one multiset, so the matrix has a single nonzero entry per row.
    """
    digits = np.indices((dim,) * level).reshape(level, -1).T
    _, col, counts = np.unique(np.sort(digits, axis=1), axis=0, return_inverse=True, return_counts=True)
    col = col.ravel()
    vals = 1 / np.sqrt(counts[col])
    return sparse.csr_array((vals, (np.arange(dim**level), col)), shape=(dim**level, comb(dim + level - 1, level)))


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


def _vec_partial_transpose_perm(dims: list[int], sys: int) -> np.ndarray:
    r"""Return `perm` such that the row-major vectorization of \(T_{sys}(X)\) is `vec(X)[perm]`."""
    num = len(dims)
    dim_total = int(np.prod(dims))
    idx = np.arange(dim_total**2).reshape(dims + dims)
    return np.swapaxes(idx, sys, num + sys).ravel()


def _apply_vec_map(vec_map: sparse.csr_array, var: cvxpy.Expression, out_dim: int) -> cvxpy.Expression:
    """Apply a linear map, given as a matrix acting on row-major vectorizations, to a square CVXPY expression."""
    return cvxpy.reshape(vec_map @ cvxpy.vec(var, order="C"), (out_dim, out_dim), order="C")


def symmetric_extension_sdp(
    dim_x: int, dim_y: int, level: int, ppt: bool = True
) -> tuple[cvxpy.Variable, cvxpy.Expression, list[cvxpy.Constraint]]:
    r"""Build a (PPT) symmetric extension of an operator on \(X \otimes Y\) on the symmetric subspace.

    A symmetric extension \(\sigma\) on \(X \otimes Y^{\otimes k}\) is supported on
    \(X \otimes \text{Sym}^k(Y)\), so it is parameterized as \(\sigma = (I_X \otimes V) Z (I_X \otimes V)^T\)
    with \(V\) the isometry onto the symmetric subspace and \(Z \succeq 0\) the (much smaller) variable.
    This removes the Bose-symmetry equality constraint. No operator on the full space is formed: the
    marginal and PPT constraints are applied directly to \(Z\) as sparse linear maps.

    PPT constraints are also reduced to the inequivalent ones. Since \(\sigma\) is invariant under
    permutations of the \(Y\) copies, transposing any single copy is unitarily equivalent to transposing
    \(Y_1\), and \(T_{Y_1}(\sigma)\) is supported on \(X \otimes Y_1 \otimes \text{Sym}^{k-1}(Y)\), where
    it is compressed. Likewise \(T_X(\sigma) \succeq 0\) if and only if \(T_X(Z) \succeq 0\).

    Args:
        dim_x: Dimension of the unextended subsystem \(X\).
        dim_y: Dimension of the extended subsystem \(Y\).
        level: Number of copies of \(Y\) in the extension.
        ppt: If `True`, require the extension to be PPT.

    Returns:
        The reduced variable \(Z\), the expression for the marginal
        \(\text{Tr}_{Y_2 \otimes \cdots \otimes Y_k}(\sigma)\) on \(X \otimes Y\), and the list of
        positivity constraints on the extension.

    """
    sym_dim = comb(dim_y + level - 1, level)
    dim_xy = dim_x * dim_y
    full_dims = [dim_x] + [dim_y] * level

    z_var = cvxpy.Variable((dim_x * sym_dim, dim_x * sym_dim), hermitian=True)
    constraints = [z_var >> 0]

    # vec(A Z A^T) = (A ⊗ A) vec(Z) maps the reduced variable to the full extension.
    embed = sparse.kron(sparse.eye_array(dim_x), _symmetric_isometry(dim_y, level), format="csr")
    embed_vec = sparse.kron(embed, embed, format="csr")
    marginal_map = _vec_partial_trace(full_dims, list(range(2, level + 1))) @ embed_vec
    marginal = _apply_vec_map(marginal_map, z_var, dim_xy)

    if ppt:
        constraints.append(partial_transpose(z_var, [0], [dim_x, sym_dim]) >> 0)
        if level > 1:
            compress = sparse.kron(sparse.eye_array(dim_xy), _symmetric_isometry(dim_y, level - 1), format="csr").T
            transpose_y1 = embed_vec[_vec_partial_transpose_perm(full_dims, 1)]
            pt_map = sparse.kron(compress, compress, format="csr") @ transpose_y1
            constraints.append(_apply_vec_map(pt_map, z_var, compress.shape[0]) >> 0)

    return z_var, marginal, constraints
