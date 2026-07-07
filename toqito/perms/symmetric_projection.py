"""Symmetric projection operator produces a projection onto a symmetric subspace."""

import math
from itertools import permutations

import numpy as np
from scipy.linalg import orth


def symmetric_projection(dim: int, p_val: int = 2, partial: bool = False) -> np.ndarray:
    r"""Produce the projection onto the symmetric subspace [@chen2014symmetric].

    For a complex Euclidean space \(\mathcal{X}\) and a positive integer \(n\), the projection onto the
    symmetric subspace is given by

    \[
        \frac{1}{n!} \sum_{\pi \in S_n} W_{\pi}
    \]

    where \(W_{\pi}\) is the swap operator and where \(S_n\) is the symmetric group on \(n\) symbols.

    Produces the orthogonal projection onto the symmetric subspace of `p_val` copies of `dim`-dimensional space.
    If `partial = True`, then the symmetric projection (PS) isn't the orthogonal projection itself, but rather a matrix
    whose columns form an orthonormal basis for the symmetric subspace (and hence the PS * PS' is the orthogonal
    projection onto the symmetric subspace).

    This function was adapted from the QETLAB package [@qetlablink].

    Args:
        dim: The dimension of the local systems.
        p_val: Default value of 2.
        partial: Default value of 0.

    Returns:
        Projection onto the symmetric subspace.

    Examples:
        The \(2\)-dimensional symmetric projection with \(p=1\) is given as \(2\)-by-\(2\) identity matrix

        \[
            \begin{pmatrix}
                1 & 0 \\
                0 & 1
            \end{pmatrix}.
        \]

        Using `|toqito⟩`, we can see this gives the proper result.

        ```python exec="1" source="above" result="text"
        from toqito.perms import symmetric_projection

        print(symmetric_projection(2, 1))
        ```


        When \(d = 2\) and \(p = 2\) we have that

        \[
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1/2 & 1/2 & 0 \\
                0 & 1/2 & 1/2 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}.
        \]

        Using `|toqito⟩` we can see this gives the proper result.

        ```python exec="1" source="above" result="text"
        from toqito.perms import symmetric_projection

        print(symmetric_projection(dim=2))
        ```

    """
    if dim < 1:
        raise ValueError("InvalidDim: `dim` must be at least 1.")
    if p_val < 1:
        raise ValueError("InvalidPVal: `p_val` must be at least 1.")

    dimp = dim**p_val

    if p_val == 1:
        return np.eye(dim)

    p_fac = math.factorial(p_val)
    sym_proj = np.zeros((dimp, dimp))

    # Accumulate the projector directly by indexing rather than summing `p_fac` dense
    # permutation operators. Each permutation `perm` of the `p_val` subsystems corresponds
    # to the permutation matrix `identity[rows, :]`, where `rows` reorders the tensor axes;
    # adding one to entry `(rows[k], k)` for every permutation builds the same sum.
    base = np.arange(dimp)
    idx = base.reshape((dim,) * p_val)
    for perm in permutations(range(p_val)):
        sym_proj[np.transpose(idx, perm).ravel(), base] += 1
    sym_proj /= p_fac

    if partial:
        sym_proj = orth(sym_proj)

    return sym_proj
