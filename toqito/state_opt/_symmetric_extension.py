r"""Shared internal helpers for symmetric-extension SDPs parameterized on the symmetric subspace.

A (Bose-)symmetric extension of an operator on \(X \otimes Y\) to \(X \otimes Y^{\otimes k}\) is supported on
\(X \otimes \mathrm{Sym}^k(Y)\), so it is parameterized as \(\sigma = (I_X \otimes V) Z (I_X \otimes V)^T\)
with \(V\) the isometry onto the symmetric subspace and \(Z \succeq 0\) a variable on the much smaller space.
Every linear map of \(\sigma\) that an SDP needs (partial traces, partial transposes, tensoring with an identity)
is a sparse matrix acting on the row-major vectorization of \(Z\), so each constraint is a single
sparse product rather than an expression tree over blocks.
"""

from math import comb

import cvxpy
import numpy as np
from scipy import sparse

from toqito.matrix_ops._vec_maps import _apply_vec_map, _vec_partial_trace, _vec_partial_transpose_perm


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


class SymmetricExtension:
    r"""A symmetric extension variable on \(X \otimes \mathrm{Sym}^k(Y)\) and the sparse maps that read it out.

    Attributes:
        variable: The Hermitian cvxpy variable \(Z\); the extension is \(\sigma = (I_X \otimes V) Z (I_X \otimes V)^T\)
            and has the same trace as \(Z\).
        dims: The dimensions ``[dim_x, dim_y, ..., dim_y]`` of the extended space, with ``level`` copies of \(Y\).
        embed: The sparse matrix mapping the vectorization of \(Z\) to that of \(\sigma\).

    """

    def __init__(self, dim_x: int, dim_y: int, level: int) -> None:
        r"""Create the variable for a symmetric extension with ``level`` copies of \(Y\)."""
        sym_dim = comb(dim_y + level - 1, level)
        self.dim_x, self.dim_y, self.level, self.sym_dim = dim_x, dim_y, level, sym_dim
        self.dims = [dim_x] + [dim_y] * level
        self.variable = cvxpy.Variable((dim_x * sym_dim, dim_x * sym_dim), hermitian=True)
        isometry = sparse.kron(sparse.eye_array(dim_x), _symmetric_isometry(dim_y, level), format="csr")
        # vec(A Z A^T) = (A (x) A) vec(Z) maps the reduced variable to the full extension.
        self.embed = sparse.kron(isometry, isometry, format="csr")

    def partial_trace(self, sys: list[int]) -> cvxpy.Expression:
        r"""Return \(\mathrm{Tr}_{sys}(\sigma)\), with subsystems numbered from \(X = 0\)."""
        vec_map = _vec_partial_trace(self.dims, sys) @ self.embed
        return _apply_vec_map(vec_map, self.variable, int(np.sqrt(vec_map.shape[0])))

    def partial_trace_map(self, sys: list[int]) -> sparse.csr_array:
        r"""Return the sparse map from the vectorization of \(Z\) to that of \(\mathrm{Tr}_{sys}(\sigma)\)."""
        return _vec_partial_trace(self.dims, sys) @ self.embed

    def partial_transpose(self, sys: list[int]) -> cvxpy.Expression:
        r"""Return \(T_{sys}(\sigma)\) on the full extended space.

        When ``sys`` is ``[0]``, transposing \(X\) commutes with the embedding and \(T_X(\sigma) \succeq 0\) exactly
        when \(T_X(Z) \succeq 0\), so the much smaller transposed \(Z\) is returned instead. When ``sys`` is ``[1]``
        the result is supported on \(X \otimes Y_1 \otimes \mathrm{Sym}^{k-1}(Y)\) and is compressed onto it.
        """
        if sys == [0]:
            perm = _vec_partial_transpose_perm([self.dim_x, self.sym_dim], 0)
            return cvxpy.reshape(cvxpy.vec(self.variable, order="C")[perm], self.variable.shape, order="C")
        vec_map = self.embed[_vec_partial_transpose_perm(self.dims, sys)]
        if sys == [1] and self.level > 1:
            compress = sparse.kron(
                sparse.eye_array(self.dim_x * self.dim_y), _symmetric_isometry(self.dim_y, self.level - 1), format="csr"
            ).T
            vec_map = sparse.kron(compress, compress, format="csr") @ vec_map
        return _apply_vec_map(vec_map, self.variable, int(np.sqrt(vec_map.shape[0])))


def symmetric_extension_sdp(
    dim_x: int, dim_y: int, level: int, ppt: bool = True
) -> tuple[cvxpy.Expression, list[cvxpy.Constraint]]:
    r"""Build a (PPT) symmetric extension of an operator on \(X \otimes Y\) on the symmetric subspace.

    PPT constraints are reduced to the inequivalent ones. Since \(\sigma\) is invariant under
    permutations of the \(Y\) copies, transposing any single copy is unitarily equivalent to transposing
    \(Y_1\), and \(T_X(\sigma) \succeq 0\) if and only if \(T_X(Z) \succeq 0\).

    Args:
        dim_x: Dimension of the unextended subsystem \(X\).
        dim_y: Dimension of the extended subsystem \(Y\).
        level: Number of copies of \(Y\) in the extension.
        ppt: If `True`, require the extension to be PPT.

    Returns:
        The expression for the marginal \(\text{Tr}_{Y_2 \otimes \cdots \otimes Y_k}(\sigma)\) on
        \(X \otimes Y\) and the list of positivity constraints on the extension.

    """
    extension = SymmetricExtension(dim_x, dim_y, level)
    constraints = [extension.variable >> 0]
    if ppt:
        constraints.append(extension.partial_transpose([0]) >> 0)
        if level > 1:
            constraints.append(extension.partial_transpose([1]) >> 0)
    return extension.partial_trace(list(range(2, level + 1))), constraints
