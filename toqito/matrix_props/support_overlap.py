"""Computes the overlap between the supports of two PSD matrices."""

import numpy as np

from toqito.matrix_ops.support_projection import support_projection


def support_overlap(mat_1: np.ndarray, mat_2: np.ndarray, tol: float = 1e-12) -> float:
    r"""Return the overlap between the supports of two PSD matrices.

    The overlap is \(\mathrm{Tr}(P_1 P_2)\), where \(P_1\) and \(P_2\) are the
    orthogonal projectors onto the supports of ``mat_1`` and ``mat_2``. It is
    zero when the supports are orthogonal and equals the dimension of the shared
    subspace when one support contains the other.

    Args:
        mat_1: A positive semidefinite matrix.
        mat_2: A positive semidefinite matrix of the same size.
        tol: Eigenvalues at or below this threshold are treated as zero when
            forming the support projectors.

    Returns:
        The overlap of the two supports.

    Examples:
        Two matrices with disjoint supports overlap by zero:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import support_overlap

        mat_1 = np.diag([1.0, 0.0])
        mat_2 = np.diag([0.0, 1.0])
        print(support_overlap(mat_1, mat_2))
        ```

    """
    proj_1 = support_projection(mat_1, tol)
    proj_2 = support_projection(mat_2, tol)
    return float(np.real_if_close(np.trace(proj_1 @ proj_2)))
