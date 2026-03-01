"""Calculates the positive semidefinite rank of a nonnegative matrix."""

import cvxpy as cp
import numpy as np

from toqito.matrix_props import is_nonnegative, is_square


def positive_semidefinite_rank(mat: np.ndarray, max_rank: int = 10) -> int | None:
    r"""Compute the positive semidefinite rank (PSD rank) of a nonnegative matrix.

    The definition of PSD rank is defined in [@fawzi2015positive].

    Finds the PSD rank of an input matrix by checking feasibility for increasing rank.

    Examples:
        As an example (Equation 21 from [@heinosaari2024can]), the PSD rank of the following matrix

        \[
            A = \frac{1}{2}
            \begin{pmatrix}
                0 & 1 & 1 \\
                1 & 0 & 1 \\
                1 & 1 & 0
            \end{pmatrix}
        \]

        is known to be \(\text{rank}_{\text{PSD}}(A) = 2\).

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import positive_semidefinite_rank

        print(positive_semidefinite_rank(1/2 * np.array([[0, 1, 1], [1,0,1], [1,1,0]])))
        ```


        The PSD rank of the identity matrix is the dimension of the matrix [@fawzi2015positive].

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import positive_semidefinite_rank

        print(positive_semidefinite_rank(np.identity(3)))
        ```

    """
    if not is_nonnegative(mat):
        raise ValueError("Matrix must be nonnegative.")
    if not is_square(mat):
        raise ValueError("Matrix must be square.")

    for k in range(1, max_rank + 1):
        if _check_psd_rank(mat, k):
            return k
    return None


def _check_psd_rank(mat: np.ndarray, max_rank: int) -> bool:
    """Check if the given PSD rank k is feasible for matrix M.

    Args:
        mat: 2D numpy ndarray.
        max_rank: The maximum rank to check.
        mat: 2D numpy ndarray
        max_rank: The maximum rank to check.

    Returns:
        True if `max_rank` is a feasible PSD rank, False otherwise.

    """
    m, n = mat.shape

    # Define variables:
    x_var = cp.Variable((m, n))

    # Define constraints:
    constraints = []
    for i in range(m):
        for j in range(n):
            constraints.append(cp.bmat([[x_var[i, j], mat[i, j]], [mat[i, j], x_var[j, i]]]) >> 0)
    constraints.append(cp.norm(x_var, "nuc") <= max_rank)

    # Define objective.
    obj = cp.sum(cp.square(x_var - mat))

    # Solve problem.
    prob = cp.Problem(cp.Minimize(obj), constraints)
    # Use CVXOPT solver (project default) which handles nuclear norm and avoids SCS compatibility issues
    prob.solve(solver=cp.CVXOPT)

    # Check if the problem is feasible and the objective is close to zero.
    return prob.status == cp.OPTIMAL and prob.value < 1e-6
