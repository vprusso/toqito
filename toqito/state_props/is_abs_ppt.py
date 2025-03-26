"""Check if a quantum state is absolutely PPT."""

import numpy as np

from toqito.matrix_props.is_density import is_density
from toqito.matrix_props.is_positive_semidefinite import is_positive_semidefinite
from toqito.state_props.in_separable_ball import in_separable_ball


def perm_inv(a):
    """Inverse of a permutation array."""
    b = [0] * len(a)
    for i, v in enumerate(a):
        b[v - 1] = i + 1
    return b


def check_ordered(X, i, j):
    """Check if matrix X is column-wise ordered at position (i, j)."""
    return not (i > 0 and X[i - 1][j] > X[i][j])


def check_cross(p, X):
    """Validate cross-ordering constraints on matrix X."""
    for i in range(p):
        for j in range(p):
            for k in range(p):
                for idx_l in range(p):
                    for m in range(p):
                        for n in range(p):
                            if (
                                X[min(i, j)][max(i, j)] > X[min(k, idx_l)][max(k, idx_l)]
                                and X[min(idx_l, n)][max(idx_l, n)] > X[min(j, m)][max(j, m)]
                                and X[min(i, n)][max(i, n)] < X[min(k, m)][max(k, m)]
                            ):
                                return False
    return True


def eigen_from_order(X, lam, dim):
    """Construct eigenvalue constraint matrix based on index matrix X."""
    p = min(dim)
    prod_dim = np.prod(dim)
    Y = np.triu(X, 1)
    X_full = np.triu(X) + np.triu(X, 1).T
    sparsity_pattern = Y > 0
    sorted_indices = np.argsort(Y[sparsity_pattern].flatten())
    Y[sparsity_pattern] = np.array(perm_inv(sorted_indices + 1))
    Y = Y + Y.T + np.diag(prod_dim + 1 - np.diag(X_full))

    L_mat = lam[prod_dim - X_full]
    L_mat += 2 * np.diag(np.diag(L_mat))
    L_mat -= lam[Y]
    return L_mat


def abs_ppt_constraints(lam, dim, escape_if_not_psd=False, lim=0):
    """Construct absolutely PPT constraint matrices from the eigenvalues.

    These matrices, if all positive semidefinite, indicate that the state is absolutely PPT.

    :param lam: Sorted eigenvalues of the density matrix.
    :param dim: Dimensions of the bipartite state.
    :param escape_if_not_psd: Stop once a PSD violation is detected.
    :param lim: Maximum number of matrices to generate.
    :return: List of constraint matrices.
    """
    lam = np.sort(np.real(lam))[::-1]
    p = min(dim)
    total = p * (p + 1) // 2
    constraints = []

    if p == 1:
        return []
    elif p == 2:
        L = np.array([[2 * lam[-1], lam[-2] - lam[0]], [lam[-2] - lam[0], 2 * lam[-3]]])
        return [L]

    X = np.full((p, p), total + 1, dtype=int)
    num_pool = [1] * total
    X[0][0] = 1
    X[0][1] = 2
    X[p - 1][p - 1] = total
    X[p - 2][p - 1] = total - 1

    def fill_matrix(i, j, low_lim):
        nonlocal constraints
        up_lim = min(j * (j + 1) // 2 + i * (p - j), total - 2)
        for k in range(low_lim, up_lim + 1):
            if num_pool[k - 1] == 1:
                X[i][j] = k
                num_pool[k - 1] = 0

                if check_ordered(X, i, j):
                    if i == p - 2 and j == p - 2:
                        if check_cross(p, X):
                            L = eigen_from_order(X, lam, dim)
                            constraints.append(L)
                            if (escape_if_not_psd and not is_positive_semidefinite(L)) or (
                                lim > 0 and len(constraints) >= lim
                            ):
                                return True
                    elif j == p - 1:
                        if fill_matrix(i + 1, i + 1, 3):
                            return True
                    elif fill_matrix(i, j + 1, k + 1):
                        return True

                num_pool[k - 1] = 1
        X[i][j] = total + 1
        return False

    fill_matrix(1, 2, 3)
    return constraints


def is_abs_ppt(rho: np.ndarray, dim: list[int] = None, max_constraints: int = 2612) -> bool | None:
    r"""Determine whether a quantum state is absolutely PPT.

    Examples
    =========
    A maximally mixed state on 2 x 2:

    >>> import numpy as np
    >>> rho = np.array([[0.5, 0.0], [0.0, 0.5]])
    >>> is_abs_ppt(rho, [2, 1])
    True

    Bell state (entangled, not absolutely PPT):

    >>> from toqito.states import bell
    >>> bell_state = bell(0) @ bell(0).conj().T
    >>> is_abs_ppt(bell_state, [2, 2])
    False

    High-dimensional identity matrix:

    >>> rho_7 = np.eye(49) / 49
    >>> is_abs_ppt(rho_7, [7, 7])
    True

    Qutrit-qutrit state (not absolutely PPT)

    >>> lam = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.04, 0.03])
    >>> is_abs_ppt(lam, [3, 3])
    False


    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param rho: A density matrix or eigenvalue vector.
    :param dim: A list of two integers representing bipartite dimensions. If `None`, assume equal split.
    :param max_constraints: Number of constraints to generate for checking.
    :raises ValueError: If `rho` is not a valid density matrix.
    :raises ValueError: If `rho` is neither a valid matrix nor a 1D eigenvalue array.
    :raises ValueError: If the product of `dim` does not match the length of `rho`.
    :return: `True` if absolutely PPT, `False` if not, or `None` if inconclusive.

    """
    rho = np.array(rho, dtype=np.complex128)

    if len(rho.shape) == 2 and rho.shape[0] != rho.shape[1]:
        raise ValueError("Input `rho` is not a valid density matrix.")

    rho = (rho + rho.T.conj()) / 2

    if len(rho.shape) == 2:
        if not is_density(rho):
            raise ValueError("Input `rho` is not a valid density matrix.")
        lam = np.sort(np.real(np.linalg.eigvalsh(rho)))[::-1]
    elif len(rho.shape) == 1:
        lam = np.sort(np.real(rho))[::-1]
    else:
        raise ValueError("Input `rho` must be a square matrix or a list of eigenvalues.")

    len_lam = len(lam)
    if dim is None:
        d = int(np.sqrt(len_lam))
        dim = [d, d]

    if len(dim) == 1:
        dim = [dim[0], len_lam // dim[0]]

    if np.prod(dim) != len_lam:
        raise ValueError("Dimensions do not match the length of `rho`.")

    p = min(dim)

    if in_separable_ball(lam):
        return True

    if np.sum(lam[: p - 1]) <= 2 * lam[-1] + np.sum(lam[-p + 1 : -1]):
        return True

    if p >= 7:
        return None

    constraints = abs_ppt_constraints(lam, dim, escape_if_not_psd=True, lim=max_constraints)
    if not constraints:
        return False

    if not is_positive_semidefinite(constraints[-1]):
        return False

    if p <= 6:
        return True

    return None
