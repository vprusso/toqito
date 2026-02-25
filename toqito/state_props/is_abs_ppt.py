"""Checks if a quantum state is absolutely PPT."""

import cvxpy as cp
import numpy as np

from toqito.matrix_props import is_hermitian, is_positive_semidefinite, is_square
from toqito.state_props.abs_ppt_constraints import abs_ppt_constraints
from toqito.state_props.in_separable_ball import in_separable_ball


def is_abs_ppt(
    mat: np.ndarray | cp.Variable, dim: int | None = None, rtol: float = 1e-05, atol: float = 1e-08
) -> bool | None | list[cp.Constraint]:
    r"""Determine whether or not a matrix is absolutely PPT [@Hildebrand_2007_AbsPPT].

    A Hermitian positive semidefinite matrix is absolutely PPT iff it is PPT under all unitary transformations.
    Equivalently, if the matrix operates on a Hilbert space \(H_{nm}\) of dimension \(nm\), then it is
    PPT under *all* possible decompositions of \(H_{nm}\) as \(H_{n} \otimes H_{m}\). Being absolutely
    PPT is a spectral condition (i.e. it is a condition on the eigenvalues of the matrix).

    The function allows passing a `numpy` ndarray or a `cvxpy` Variable for `mat`:

    - If `mat` is a `numpy` ndarray, the function first checks if `mat` is Hermitian positive
        semidefinite. Then, it checks if its eigenvalues satisfy the Gerschgorin circle property (see Theorem 7.2 of
        [@Jivulescu_2015_Reduction]). Then it checks if the matrix belongs to the separable ball by calling
        `in_separable_ball`. Finally, if all the above checks fail to return a definite result, it determines if
        the matrix is absolutely PPT by checking if all the constraint matrices returned by
        `abs_ppt_constraints` are positive semidefinite.
    - If `mat` is a `cvxpy` Variable, `mat` must be a 1D vector representing the eigenvalues of
        a matrix. The function then returns the list of `cvxpy` Constraints required for optimizing over the
        space of absolutely PPT matrices. This includes the positive semidefinite constraint on each constraint matrix
        as well as `mat[0] ≥ mat[1] ≥ ... ≥ mat[-1] ≥ 0`.

    This function is adapted from QETLAB [@QETLAB_link].

    !!! Note
        If `min(dim)` \(\leq 6\), this function checks all constraints
        and therefore returns `True` or `False` in all cases. However, if
        `min(dim)` \(\geq 7\), only the first \(33592\) constraints are
        checked, since there are over \(23178480\) constraints in this case
        [@Johnston_2014_Orderings]. Therefore the function returns either
        `False` if at least one constraint was not satisfied, or `None`
        if all checked constraints were satisfied.

    Examples:
    A random density matrix will likely not be absolutely PPT:

    ```python exec="1" source="above"
    import numpy as np
    from toqito.rand import random_density_matrix
    from toqito.state_props import is_abs_ppt
    rho = random_density_matrix(9) # assumed to act on a 3 x 3 bipartite system
    print(f"ρ is absolutely PPT: {is_abs_ppt(rho, 3)}")
    ```

    The maximally-mixed state is an example of an absolutely PPT state:

    ```python exec="1" source="above"
    import numpy as np
    from toqito.states import max_mixed
    from toqito.state_props import is_abs_ppt
    rho = max_mixed(9) # assumed to act on a 3 x 3 bipartite system
    print(f"ρ is absolutely PPT: {is_abs_ppt(rho, 3)}")
    ```

    Raises:
        TypeError: If `mat` is not a `numpy` ndarray or a `cvxpy` Variable.
        ValueError: If `mat` is a `numpy` ndarray but is not square.
        ValueError: If `mat` is a `cvxpy` Variable but is not 1D.
        ValueError: If `dim` does not divide the dimensions of `mat`.

    Args:
        mat: A square matrix.
        dim: The dimension of any one subsystem on which `mat` acts. If `None`, `dim` is selected such that `min(dim, mat.shape[0] // dim)` is maximised, since this gives the strongest conditions on being absolutely PPT (see Theorem 2 of [@Hildebrand_2007_AbsPPT]).
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        If `mat` is a 1D `cvxpy` Variable, return a list of `cvxpy` Constraints required for optimizing over the space of absolutely PPT matrices.

    """
    if isinstance(mat, np.ndarray):
        if not is_square(mat):
            raise ValueError(f"Expected mat to be square: however mat.shape was {mat.shape}")
    elif isinstance(mat, cp.Variable):
        if mat.ndim != 1:
            raise ValueError(f"Expected mat to be 1D: however mat had {mat.ndim} dimensions")
    else:
        raise TypeError("mat must be a square numpy ndarray or a 1D cvxpy Variable")

    nm = mat.shape[0]

    if dim is None:
        # Find the largest divisor d of nm such that d ** 2 <= nm.
        # nm won't be too large, so let's just use a for-loop.
        # Floating-point arithmetic is risky.
        dim, j = 1, 1
        while True:
            if j**2 > nm:
                break
            if nm % j == 0:
                dim = j
            j += 1

    if nm % dim != 0:
        raise ValueError("Calculated subsystem dimensions and provided matrix dimensions are incompatible")

    n, m = dim, nm // dim
    p = min(n, m)

    if isinstance(mat, np.ndarray):
        # Quick checks:
        # 1. Check if mat is Hermitian.
        if not is_hermitian(mat, rtol, atol):
            return False
        # Compute eigenvalues (in descending order).
        # np.linalg.eigsvalsh normally returns eigenvalues in ascending order,
        # but it is risky to assume this will remain the default behaviour in the future.
        eigs = np.sort(np.linalg.eigvalsh(mat))[::-1]
        # 2. Check if mat is PSD.
        if eigs[-1] < -abs(atol):
            return False
        # 3. Check Theorem 7.2 of [@Jivulescu_2015_Reduction].
        if sum(eigs[: p - 1]) <= eigs[-1] + sum(eigs[-p:]):
            return True
        # 4. Check if mat is in separable ball.
        if in_separable_ball(mat):
            return True
        # All quick checks failed, so construct constraint matrices and check if all are PSD.
        constraints = abs_ppt_constraints(eigs, p)
        for constraint in constraints:
            if not is_positive_semidefinite(constraint, rtol, atol):
                return False
        # We checked all constraints for p <= 6, but not for p >= 7.
        return True if p <= 6 else None
    else:
        constraints = abs_ppt_constraints(mat, p, use_check=True)
        return [mat[-1] >= 0] + [mat[i] >= mat[i + 1] for i in range(nm - 1)] + [c_mat >> 0 for c_mat in constraints]
