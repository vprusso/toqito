"""Checks if a quantum state is absolutely PPT (PPT from spectrum)."""

import numpy as np

from toqito.matrix_props import is_hermitian, is_positive_semidefinite, is_square


def _abs_ppt_constraints(eigvals: np.ndarray, dim: list[int]) -> list[np.ndarray]:
    r"""Generate the constraint matrices for the absolutely PPT test, following QETLAB/AbsPPTConstraints.

    :cite:`Hildebrand_2007_PPT, QETLAB_link`.

    Only implemented for 2x2, 2x3, and 3x3 systems.
    """
    eigvals = np.sort(np.real(eigvals))[::-1]
    d1, d2 = dim
    constraints = []

    if (d1, d2) == (2, 2):
        # Only one constraint: 2x2 matrix
        # See Hildebrand, Eq. (2)
        # Matrix: [[2*lam4, lam3-lam1], [lam3-lam1, 2*lam2]]
        L = np.array(
            [
                [2 * eigvals[3], eigvals[2] - eigvals[0]],
                [eigvals[2] - eigvals[0], 2 * eigvals[1]],
            ]
        )
        constraints.append(L)
    elif (d1, d2) in [(2, 3), (3, 2)]:
        # Two constraints: 3x3 matrices
        # See Hildebrand, Eq. (3)
        # eigvals: lam1 >= ... >= lam6
        L1 = np.array(
            [
                [2 * eigvals[5], eigvals[4] - eigvals[0], eigvals[3] - eigvals[1]],
                [eigvals[4] - eigvals[0], 2 * eigvals[2], eigvals[1] - eigvals[3]],
                [eigvals[3] - eigvals[1], eigvals[1] - eigvals[3], 2 * eigvals[0]],
            ]
        )
        L2 = np.array(
            [
                [2 * eigvals[5], eigvals[4] - eigvals[0], eigvals[2] - eigvals[1]],
                [eigvals[4] - eigvals[0], 2 * eigvals[3], eigvals[1] - eigvals[2]],
                [eigvals[2] - eigvals[1], eigvals[1] - eigvals[2], 2 * eigvals[0]],
            ]
        )
        constraints.extend([L1, L2])
    elif (d1, d2) == (3, 3):
        # Two constraints: 3x3 matrices
        # See QETLAB/AbsPPTConstraints and Hildebrand, Eq. (4)
        # eigvals: lam1 >= ... >= lam9
        L1 = np.array(
            [
                [2 * eigvals[8], eigvals[7] - eigvals[0], eigvals[6] - eigvals[1]],
                [eigvals[7] - eigvals[0], 2 * eigvals[5], eigvals[4] - eigvals[2]],
                [eigvals[6] - eigvals[1], eigvals[4] - eigvals[2], 2 * eigvals[3]],
            ]
        )
        L2 = np.array(
            [
                [2 * eigvals[8], eigvals[7] - eigvals[0], eigvals[5] - eigvals[1]],
                [eigvals[7] - eigvals[0], 2 * eigvals[6], eigvals[4] - eigvals[2]],
                [eigvals[5] - eigvals[1], eigvals[4] - eigvals[2], 2 * eigvals[3]],
            ]
        )
        constraints.extend([L1, L2])
    else:
        # Not implemented for higher dimensions
        raise NotImplementedError("Absolutely PPT constraints for dimensions > 3x3 are not implemented.")
    return constraints


def is_abs_ppt(mat: np.ndarray, dim: None | int | list[int] = None, atol: float = 1e-8) -> bool | None:
    r"""Determine whether a density matrix is absolutely PPT (PPT from spectrum).

    This function checks whether a density matrix is absolutely PPT using the Hildebrand constraints
    for 2x2, 2x3, and 3x3 bipartite systems :cite:`Hildebrand_2007_PPT`
    and the reference implementation from QETLAB :cite:`QETLAB_link`.

    Note:
        The Hildebrand/QETLAB constraints are only known for 2x2, 2x3, and 3x3 bipartite systems.
        For other dimensions, the absolutely PPT property is undecidable with this method and the function returns None.

    Examples
    ==========
    Consider the maximally mixed state in 2x2 dimensions:

    .. jupyter-execute::

        from toqito.states import max_mixed
        from toqito.state_props import is_abs_ppt
        rho = max_mixed(4)
        is_abs_ppt(rho, [2, 2])

    :param mat: The density matrix to check.
    :param dim: The local dimensions (default: inferred as equal dims).
    :param atol: Numerical tolerance for positive semidefinite checks.
    :return: :code:`True` if absolutely PPT, :code:`False` if not, :code:`None` if undecidable (for large dims).
    :raises ValueError: If the input matrix is not square, not Hermitian, does not have trace 1, is not positive
        semidefinite, or if the provided dimensions do not match the matrix size.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """
    if not is_square(mat):
        raise ValueError("Input matrix must be square.")
    if not is_hermitian(mat, atol=atol):
        raise ValueError("Input matrix must be Hermitian.")
    if not is_positive_semidefinite(mat, atol=atol):
        raise ValueError("Input matrix must be positive semidefinite.")
    if not np.isclose(np.trace(mat), 1, atol=atol):
        raise ValueError("Input matrix must have trace 1 (density matrix).")

    n = mat.shape[0]
    # Infer dimensions if not provided
    if dim is None:
        # Try to factor n as d1*d2 with d1 <= d2
        d1 = int(np.floor(np.sqrt(n)))
        while n % d1 != 0 and d1 > 1:
            d1 -= 1
        d2 = n // d1
        dim = [d1, d2]
    elif isinstance(dim, int):
        d1 = dim
        d2 = n // d1
        dim = [d1, d2]
    else:
        d1, d2 = dim
    if d1 * d2 != n:
        raise ValueError(f"Dimensions {d1} x {d2} do not match matrix size {n}.")

    # Only implemented for 2x2, 2x3, 3x2, 3x3 (see Hildebrand 2007, QETLAB)
    if (d1, d2) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        eigvals = np.linalg.eigvalsh(mat)
        eigvals = np.sort(np.real(eigvals))[::-1]
        constraints = _abs_ppt_constraints(eigvals, [d1, d2])
        for L in constraints:
            if not is_positive_semidefinite(L, atol=atol):
                return False
        return True
    # For higher dimensions, use SDP approach as in QETLAB
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy is required for the SDP-based absolutely PPT test in higher dimensions.")

    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.sort(np.real(eigvals))[::-1]
    n = d1 * d2
    # Variable: Hermitian density matrix with same spectrum as mat
    rho_var = cp.Variable((n, n), hermitian=True)
    constraints = [
        rho_var >> 0,
        cp.trace(rho_var) == 1,
    ]
    # Spectrum constraint: the eigenvalues of rho_var majorized by eigvals and vice versa (Schur-Horn)
    # Enforce that the spectrum of rho_var is exactly eigvals (up to tolerance)
    # This is done by constraining the sum of the k largest eigenvalues
    for k in range(1, n + 1):
        constraints.append(cp.lambda_sum_largest(rho_var, k) <= np.sum(eigvals[:k]) + atol)
        constraints.append(cp.lambda_sum_largest(rho_var, k) >= np.sum(eigvals[:k]) - atol)
    # Partial transpose constraint: rho_var^{T_B} >= 0
    def partial_transpose(mat, dims, sys=1):
        # sys=1 means partial transpose on second subsystem
        d1, d2 = dims
        mat = cp.reshape(mat, (d1, d2, d1, d2))
        mat = cp.transpose(mat, (0, 3, 2, 1))
        return cp.reshape(mat, (n, n))
    rho_pt = partial_transpose(rho_var, [d1, d2])
    constraints.append(rho_pt >> 0)
    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception:
        return None
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return True
    elif prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
        return False
    else:
        return None


