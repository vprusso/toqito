"""Checks whether the support of one PSD matrix is contained in another's."""

import numpy as np

from toqito.matrix_ops.support_projection import support_projection


def is_support_subset(mat_1: np.ndarray, mat_2: np.ndarray, tol: float = 1e-12) -> bool:
    r"""Check whether the support of ``mat_1`` is contained in that of ``mat_2``.

    The support of ``mat_1`` is contained in the support of ``mat_2`` when no
    part of \(P_1\) leaks outside \(P_2\), i.e. when
    \(\mathrm{Tr}\!\left(P_1 (I - P_2)\right) \le\) ``tol``, where \(P_1\) and
    \(P_2\) are the support projectors of ``mat_1`` and ``mat_2``.

    Args:
        mat_1: A positive semidefinite matrix.
        mat_2: A positive semidefinite matrix of the same size.
        tol: Eigenvalues at or below this threshold are treated as zero when
            forming the support projectors, and the same threshold bounds the
            allowed leakage.

    Returns:
        ``True`` if the support of ``mat_1`` is contained in that of ``mat_2``.

    Examples:
        The support of a rank-one matrix sits inside the full-rank identity:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_support_subset

        mat_1 = np.diag([1.0, 0.0])
        print(is_support_subset(mat_1, np.eye(2)))
        ```

    """
    proj_1 = support_projection(mat_1, tol)
    proj_2 = support_projection(mat_2, tol)
    leak = np.trace(proj_1 @ (np.eye(mat_1.shape[0]) - proj_2))
    return float(np.real_if_close(leak)) <= tol
