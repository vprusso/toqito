"""Determine the factor width of a positive semidefinite matrix."""

from collections import deque
from typing import Any

import cvxpy as cp
import cvxpy.settings as cp_settings
import numpy as np
import scipy.sparse as sp

from toqito.matrix_ops import null_space
from toqito.matrix_props import is_positive_semidefinite


def factor_width(
    mat: np.ndarray,
    k: int,
    *,
    solver: str | None = "SCS",
    solver_kwargs: dict | None = None,
    tol: float = 1e-8,
) -> dict:
    r"""Decide whether a positive semidefinite matrix has factor width at most :math:`k`.

    The factor width of a matrix is the minimal value of :math:`k` for which it
    admits a decomposition :math:`M = \sum_j v_j v_j^*` with each :math:`v_j`
    supported on at most :math:`k` coordinates.  This routine implements the
    low-rank algorithm in :footcite:`Johnston_2025_Complexity`.

    Examples
    ========

    The matrix :math:`\operatorname{diag}(1, 1, 0)` has factor width at most :math:`1`.

    .. jupyter-execute::

        import numpy as np
        from toqito.matrix_props import factor_width

        diag_mat = np.diag([1, 1, 0])
        result = factor_width(diag_mat, k=1)
        result["feasible"]

    Conversely, the rank-one matrix :math:`\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}/2` is not
    :math:`1`-factorable.

    .. jupyter-execute::

        import numpy as np
        from toqito.matrix_props import factor_width

        hadamard = np.array([[1, 1], [1, 1]], dtype=np.complex128) / 2
        result = factor_width(hadamard, k=1)
        result["feasible"]

    :param mat: Positive semidefinite matrix to test.
    :param k: Target factor width bound.
    :param solver: CVXPY solver name (defaults to :code:`"SCS"`).
    :param solver_kwargs: Additional keyword arguments forwarded to
        :meth:`cvxpy.Problem.solve`.
    :param tol: Numerical tolerance used for rank computations and duplicate detection.
    :return: Dictionary with keys
        ``feasible`` (boolean flag),
        ``status`` (solver status string),
        ``factors`` (list of PSD matrices whose sum equals ``mat`` when feasible), and
        ``subspaces`` (orthonormal bases spanning the subspaces used in the decomposition).

    """
    mat = np.asarray(mat, dtype=np.complex128)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix must be square.")

    d = mat.shape[0]
    if k < 1 or k > d:
        raise ValueError("The factor width parameter k must satisfy 1 <= k <= d.")

    if not is_positive_semidefinite(mat, atol=tol):
        raise ValueError("Input matrix must be positive semidefinite.")

    if k == d:
        return {
            "feasible": True,
            "status": "trivial",
            "factors": [mat],
            "subspaces": [np.eye(d, dtype=np.complex128)],
        }

    # Obtain an orthonormal basis for the range of mat.
    eig_vals, eig_vecs = np.linalg.eigh((mat + mat.conj().T) / 2)
    support = eig_vals > tol
    if not np.any(support):
        return {
            "feasible": True,
            "status": "trivial",
            "factors": [np.zeros_like(mat)],
            "subspaces": [np.zeros((d, 0), dtype=np.complex128)],
        }
    range_basis = eig_vecs[:, support]

    max_zero_count = d - k
    subspaces = _enumerate_support_subspaces(range_basis, max_zero_count, tol)
    if not subspaces:
        return {
            "feasible": False,
            "status": "no_support_subspace",
            "factors": None,
            "subspaces": [],
        }

    # Build the SDP: variables live in the reduced coordinates of each subspace.
    mat_block = _complex_to_real_block(mat)

    variables: list[tuple[np.ndarray, cp.Variable]] = []
    components = []
    constraints = []

    for basis in subspaces:
        dim = basis.shape[1]
        if dim == 0:
            continue
        basis_block = _complex_to_real_block(basis)
        var = cp.Variable((2 * dim, 2 * dim), PSD=True)
        lift_block = basis_block @ var @ basis_block.T
        variables.append((basis, var))
        components.append(lift_block)

    if not components:
        return {
            "feasible": False,
            "status": "no_support_subspace",
            "factors": None,
            "subspaces": [],
        }

    total = components[0]
    for comp in components[1:]:
        total += comp
    constraints.append(total == mat_block)
    problem = cp.Problem(cp.Minimize(cp.Constant(0)), constraints)

    status = _solve_problem(problem, solver, solver_kwargs)

    feasible = status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    factor_matrices = None
    if feasible:
        factor_matrices = []
        for basis, var in variables:
            if var.value is None:
                return {
                    "feasible": False,
                    "status": status,
                    "factors": None,
                    "subspaces": subspaces,
                }
            local_factor = _real_block_to_complex(var.value)
            factor_matrices.append(basis @ local_factor @ basis.conj().T)

    return {
        "feasible": feasible,
        "status": status,
        "factors": factor_matrices,
        "subspaces": subspaces,
    }


def _solve_problem(
    problem: cp.Problem,
    solver: str | None,
    solver_kwargs: dict[str, Any] | None,
) -> str:
    """Solve a CVXPY problem and return the solver status."""
    solve_kwargs = dict(solver_kwargs or {})

    if _is_scs_solver(solver):
        return _solve_problem_with_scs(problem, solve_kwargs)

    if solver is None:
        problem.solve(**solve_kwargs)
    else:
        problem.solve(solver=solver, **solve_kwargs)
    return problem.status


def _solve_problem_with_scs(
    problem: cp.Problem,
    solver_kwargs: dict[str, Any],
) -> str:
    """Solve with SCS ensuring sparse matrices are provided in CSC form."""
    warm_start = bool(solver_kwargs.pop("warm_start", False))
    verbose = bool(solver_kwargs.pop("verbose", False))

    data, chain, inverse_data = problem.get_problem_data(cp.SCS)
    for key in (cp_settings.A, cp_settings.P):
        if key in data and data[key] is not None:
            data[key] = sp.csc_matrix(data[key])

    solution = chain.solve_via_data(
        problem,
        data,
        warm_start=warm_start,
        verbose=verbose,
        solver_opts=solver_kwargs,
    )
    problem.unpack_results(solution, chain, inverse_data)
    return problem.status


def _is_scs_solver(solver: Any | None) -> bool:
    """Return ``True`` when the solver selection corresponds to SCS."""
    if solver is None:
        return False
    if solver is cp.SCS:
        return True
    if isinstance(solver, str) and solver.strip().upper() == "SCS":
        return True
    return False


def _complex_to_real_block(mat: np.ndarray) -> np.ndarray:
    """Embed a complex matrix into its real block representation."""
    real = mat.real
    imag = mat.imag
    top = np.concatenate((real, -imag), axis=1)
    bottom = np.concatenate((imag, real), axis=1)
    return np.concatenate((top, bottom), axis=0)


def _real_block_to_complex(block: np.ndarray) -> np.ndarray:
    """Recover a complex matrix from its real block representation."""
    rows, cols = block.shape
    if rows % 2 != 0 or cols % 2 != 0:
        raise ValueError("Real block matrix must have even dimensions.")
    half_rows = rows // 2
    half_cols = cols // 2
    real = block[:half_rows, :half_cols]
    imag = block[half_rows:, :half_cols]
    return real + 1j * imag


def _enumerate_support_subspaces(
    range_basis: np.ndarray,
    max_zero_count: int,
    tol: float,
) -> list:
    """Enumerate the unique subspaces obtained by zeroing coordinates."""
    d, _ = range_basis.shape
    queue: deque[tuple[frozenset[int], np.ndarray]] = deque()
    queue.append((frozenset(), range_basis))

    seen: dict[bytes, dict[str, int | np.ndarray]] = {}
    result: list = []

    while queue:
        zero_set, basis = queue.popleft()
        key, ortho_basis = _canonical_key(basis, tol)
        entry = seen.get(key)
        if entry is None:
            seen[key] = {"basis": ortho_basis, "max_zero": len(zero_set)}
            entry = seen[key]
        else:
            entry["max_zero"] = max(entry["max_zero"], len(zero_set))

        if len(zero_set) < max_zero_count:
            for idx in range(d):
                if idx in zero_set:
                    continue
                new_zero_set = frozenset(set(zero_set) | {idx})
                if len(new_zero_set) > max_zero_count:
                    continue
                new_basis = _intersect_with_zero(entry["basis"], idx, tol)
                if new_basis.shape[1] == 0:
                    continue
                queue.append((new_zero_set, new_basis))

    for entry in seen.values():
        basis = entry["basis"]
        nonzero_support = _max_support_size(basis, tol)
        if nonzero_support <= (basis.shape[0] - max_zero_count) and entry["max_zero"] >= max_zero_count:
            result.append(basis)

    return result


def _canonical_key(basis: np.ndarray, tol: float) -> tuple[bytes, np.ndarray]:
    """Return a hashable key and orthonormal basis for a subspace."""
    if basis.size == 0:
        zero_basis = np.zeros((basis.shape[0], 0), dtype=np.complex128)
        return b"zero", zero_basis
    q, _ = np.linalg.qr(basis)
    projector = q @ q.conj().T
    rounded = np.round(projector.real, decimals=8) + 1j * np.round(projector.imag, decimals=8)
    return rounded.tobytes(), q


def _intersect_with_zero(basis: np.ndarray, index: int, tol: float) -> np.ndarray:
    """Intersect the subspace spanned by ``basis`` with the hyperplane ``v_index = 0``."""
    if basis.shape[1] == 0:
        return basis
    row = basis[index, :].reshape(1, -1)
    kernel = null_space(row, tol=tol)
    if kernel.size == 0:
        return np.zeros((basis.shape[0], 0), dtype=np.complex128)
    intersection = basis @ kernel
    if intersection.size == 0:
        return np.zeros((basis.shape[0], 0), dtype=np.complex128)
    q, _ = np.linalg.qr(intersection)
    return q


def _max_support_size(basis: np.ndarray, tol: float) -> int:
    """Compute the maximum support size of vectors in the span of ``basis``."""
    if basis.shape[1] == 0:
        return 0
    # The support size is bounded by the number of indices not identically zero.
    mask = np.linalg.norm(basis, axis=1) > tol
    return int(np.count_nonzero(mask))

