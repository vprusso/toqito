"""Checks if a quantum state is absolutely PPT (PPT from spectrum)."""

import numpy as np

from toqito.matrix_props import is_hermitian, is_positive_semidefinite, is_square


def _abs_ppt_constraints(eigvals: np.ndarray, dim: list[int]) -> list[np.ndarray]:
    r"""Generate the constraint matrices for the absolutely PPT test.

    The Hildebrand constraints are a set of necessary and sufficient conditions for a state to be absolutely PPT
    (PPT from spectrum) in 2x2, 2x3, and 3x3 bipartite systems. These constraints are derived from the fact that
    a state is absolutely PPT if and only if its partial transpose remains positive semidefinite under any unitary
    transformation that preserves its spectrum.

    For 2x2 systems, there is a single constraint matrix:
    [[2*lam4, lam3-lam1], [lam3-lam1, 2*lam2]]

    For 2x3 and 3x2 systems, there are two constraint matrices that must be positive semidefinite.

    For 3x3 systems, there are also two constraint matrices that must be checked.

    These constraints were first derived in :cite:`Hildebrand_2007_PPT` and implemented in QETLAB.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames
        :style: plain

    :param eigvals: The sorted eigenvalues of the density matrix in descending order
    :param dim: The dimensions of the bipartite system [d1, d2]
    :return: List of constraint matrices that must be positive semidefinite
    :raises NotImplementedError: If dimensions > 3x3 are provided

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

    A quantum state is said to be absolutely PPT (PPT from spectrum) if its partial transpose remains positive
    semidefinite under any unitary transformation that preserves its spectrum. This property is stronger than
    being PPT, as it depends only on the eigenvalues of the state.

    This function implements the necessary and sufficient conditions for absolute PPT derived by Hildebrand
    for 2x2, 2x3, and 3x3 bipartite systems. For higher dimensions, it returns None as the property is
    undecidable with the current method.

    The implementation follows the approach from QETLAB's IsAbsPPT function.

    Examples
    ==========
    Consider the maximally mixed state in 2x2 dimensions:

    .. jupyter-execute::
        from toqito.states import max_mixed
        from toqito.state_props import is_abs_ppt
        rho = max_mixed(4)
        is_abs_ppt(rho, [2, 2])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames
        :style: plain

    :param mat: The density matrix to check
    :param dim: The local dimensions (default: inferred as equal dims)
    :param atol: Numerical tolerance for positive semidefinite checks
    :return: True if absolutely PPT, False if not, None if undecidable (for large dims)
    :raises ValueError: If the input matrix is not square, not Hermitian, does not have trace 1,
        is not positive semidefinite, or if the provided dimensions do not match the matrix size

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

    # For higher dimensions, we cannot determine if the state is absolutely PPT
    # using the current method, as the necessary and sufficient conditions are
    # only known for 2x2, 2x3, and 3x3 systems
    return None
