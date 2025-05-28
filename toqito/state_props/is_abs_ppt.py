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


def is_abs_ppt(mat: np.ndarray, dim: None | int | list[int] = None, tol: float = 1e-8) -> bool | int:
    r"""Determine whether a density matrix is absolutely PPT (PPT from spectrum).

    :cite:`Hildebrand_2007_PPT, QETLAB_link`.

    A density matrix is absolutely PPT if every unitary conjugation of it is PPT. This is a spectrum-based
    property and can be checked using the Hildebrand/QETLAB constraints for small dimensions.

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
    :param tol: Numerical tolerance for positive semidefinite checks.
    :return: :code:`True` if absolutely PPT, :code:`False` if not, :code:`-1` if undecidable (for large dims).
    :raises ValueError: If the input matrix is not square, not Hermitian, does not have trace 1, is not positive
        semidefinite, or if the provided dimensions do not match the matrix size.

    Note:
        Returns -1 if the absolutely PPT property is undecidable for the given dimensions (i.e., for dimensions
        other than 2x2, 2x3, 3x2, or 3x3).

    """
    if not is_square(mat):
        raise ValueError("Input matrix must be square.")
    if not is_hermitian(mat, atol=tol):
        raise ValueError("Input matrix must be Hermitian.")
    if not is_positive_semidefinite(mat, atol=tol):
        raise ValueError("Input matrix must be positive semidefinite.")
    if not np.isclose(np.trace(mat), 1, atol=tol):
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

    # Only implemented for 2x2, 2x3, 3x2, 3x3
    if (d1, d2) not in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        return -1  # undecidable for now

    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.sort(np.real(eigvals))[::-1]
    constraints = _abs_ppt_constraints(eigvals, [d1, d2])
    for L in constraints:
        if not is_positive_semidefinite(L, atol=tol):
            return False
    return True


