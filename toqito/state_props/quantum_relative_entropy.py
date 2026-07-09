r"""Quantum relative entropy for positive semidefinite matrices."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import contextlib
import warnings
from typing import Any

import cvxpy
import numpy as np
from scipy.linalg import logm

from toqito.cones._utils import _require_square_2d
from toqito.cones.integral_relative_entropy import evaluate_relative_entropy_integral
from toqito.cones.ln_quantum_entropy import ln_quantum_entropy
from toqito.cones.operator_relative_entropy_epi_cone import (
    operator_relative_entropy_epi_cone,
)
from toqito.cones.trace_matrix_log import trace_matrix_log
from toqito.matrix_props import is_hermitian, is_positive_semidefinite


@contextlib.contextmanager
def _silence_singular_logm_warning():
    """Silence scipy's cosmetic "logm input matrix is exactly singular" warning.

    The numeric relative-entropy paths evaluate ``scipy.linalg.logm`` (directly and via
    ``ln_quantum_entropy``/``trace_matrix_log``) on rank-deficient density matrices, which is
    routine for pure states and for the reduced/lifted operators used by callers such as
    ``quantum_conditional_entropy``. The singular directions contribute the correct
    ``0 * (-inf) = 0`` to the trace, so the returned value is valid and the warning is noise. The
    filter matches only that exact message, so genuine warnings still propagate.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The logm input matrix is exactly singular.")
        yield


def quantum_relative_entropy(
    mat_x: np.ndarray | cvxpy.Expression,
    mat_y: np.ndarray | cvxpy.Expression,
    m: int = 3,
    k: int = 3,
    apx: int = 0,
    space_optimized: bool = False,
    epsilon_dec: float = 1e-2,
    solver: str = "SCS",
    **solve_kwargs: Any,
) -> float:
    r"""Compute the quantum relative entropy \(D(X||Y)\) for PSD \(X\) and \(Y\).

    Note that this function uses the natural logarithm (base e) for the
    entropy calculation and not the base 2 logarithm.

    The quantum relative entropy is jointly convex in \(X\) and \(Y\), assuming
    that \(X\) and \(Y\) are positive semidefinite matrices.

    This function includes two modes for the affine branch. The first (``space_optimized=False``) is from
    [@fawzi2017matrixlogarithm] and the second (``space_optimized=True``) is
    from [@kossmann2024optimisingrelativeentropy].

    The first mode features better convergence in the approximation parameters,
    but it relies on a lifting technique that makes variables of size \(n^2 \times n^2\) instead of \(n \times n\).
    The second mode uses a smaller semidefinite representation and is more efficient, but it may not be as accurate.
    To increase accuracy with the second mode, you can lower the ``epsilon_dec`` parameter.

    Args:
        mat_x: The first positive semidefinite matrix.
        mat_y: The second positive semidefinite matrix.
        m: The number of quadrature nodes to use. Ignored when ``space_optimized`` is
            ``True`` and both arguments are non-constant affine CVXPY expressions.
        k: The number of square-roots to take. Ignored when ``space_optimized`` is
            ``True`` and both arguments are non-constant affine CVXPY expressions.
        apx: The approximation to use. Ignored when ``space_optimized`` is ``True`` and
            both arguments are non-constant affine CVXPY expressions.
        space_optimized: If ``True``, use the integral SDP representation
            [@kossmann2024optimisingrelativeentropy] when both arguments are
            non-constant affine CVXPY expressions. This uses \(n \times n\)
            matrix variables instead of the Kronecker \(n^2 \times n^2\) cone
            from CVXQUAD.
        epsilon_dec: Grid refinement parameter for the integral representation.
            Used only when ``space_optimized`` is ``True``. Defaults to ``1e-2``.
        solver: CVXPY solver for the joint-affine SDP branches. Defaults to ``"SCS"``.
        solve_kwargs: Additional arguments passed to ``cvxpy.Problem.solve`` for the
            joint-affine SDP branches.

    Raises:
        ValueError: If ``mat_x`` or ``mat_y`` is not a numpy array or cvxpy expression.
        ValueError: If the input matrices are not square or 2D.
        ValueError: If the input matrices are not positive semidefinite.
        ValueError: If the input matrices are not Hermitian.
        ValueError: If the number of quadrature nodes is not a positive integer.
        ValueError: If the number of square-roots is not a positive integer.
        ValueError: If the approximation is not -1, 0, or 1.
        ValueError: If ``mat_x`` and ``mat_y`` do not have the same shape.
        ValueError: If ``space_optimized`` is not a boolean.

    Returns:
        The quantum relative entropy \(D(X||Y)\) as a float.

    Examples:
        Compute \(D(|0\rangle\langle 0| \parallel I / 2)\), which is
        \(\ln(2)\) in nats.

        ```python exec="1" source="above" result="text"
        import numpy as np

        from toqito.state_props import quantum_relative_entropy

        rho = np.array([[1.0, 0.0], [0.0, 0.0]])
        sigma = np.identity(2) / 2
        print(f"{quantum_relative_entropy(rho, sigma):.6f}")
        ```

    """
    if not isinstance(mat_x, (np.ndarray, cvxpy.Expression)):
        raise ValueError("mat_x must be a numpy array or a cvxpy expression")
    if not isinstance(mat_y, (np.ndarray, cvxpy.Expression)):
        raise ValueError("mat_y must be a numpy array or a cvxpy expression")
    _require_square_2d(mat_x, "mat_x")
    _require_square_2d(mat_y, "mat_y")
    if mat_x.shape != mat_y.shape:
        raise ValueError("mat_x and mat_y must have the same shape")

    if not m > 0:
        raise ValueError("m must be a positive integer")
    if not k > 0:
        raise ValueError("k must be a positive integer")
    if apx not in [-1, 0, 1]:
        raise ValueError("apx must be -1, 0, or 1")
    if space_optimized not in [True, False]:
        raise ValueError("space_optimized must be a boolean")

    if isinstance(mat_x, np.ndarray) and isinstance(mat_y, np.ndarray):
        tol = 1e-9
        if not is_hermitian(mat_x):
            raise ValueError("mat_x must be a Hermitian matrix")
        if not is_hermitian(mat_y):
            raise ValueError("mat_y must be a Hermitian matrix")
        if not is_positive_semidefinite(mat_x):
            raise ValueError("mat_x must be a positive semidefinite matrix")
        if not is_positive_semidefinite(mat_y):
            raise ValueError("mat_y must be a positive semidefinite matrix")

        mat_x = (mat_x + mat_x.conj().T) / 2
        mat_y = (mat_y + mat_y.conj().T) / 2
        eig_x, vec_x = np.linalg.eigh(mat_x)
        eig_y, vec_y = np.linalg.eigh(mat_y)
        overlaps = vec_x.conj().T @ vec_y
        u = eig_x @ (np.abs(overlaps) ** 2)

        if np.any(u[eig_y <= tol] > tol):
            raise ValueError("D(X||Y) is infinity because im(X) is not contained in im(Y)")
        else:
            r1 = np.sum(eig_x[eig_x > tol] * np.log(eig_x[eig_x > tol]))
            r2 = np.sum(u[eig_y > tol] * np.log(eig_y[eig_y > tol]))
            return float(r1 - r2)
    elif isinstance(mat_x, cvxpy.Expression) and mat_x.is_constant():
        if mat_x.value is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_x as a numpy.ndarray."
            )
        x_val = np.asarray(mat_x.value)
        # tr(A log B): trace_matrix_log(X, C) = tr(C log X) with X = B, C = A.
        if isinstance(mat_y, np.ndarray):
            mat_y_log = np.asarray(mat_y)
        elif isinstance(mat_y, cvxpy.Expression) and mat_y.is_constant():
            if mat_y.value is None:
                raise ValueError("mat_y has no numeric value; pass a numpy.ndarray or set `.value`.")
            mat_y_log = np.asarray(mat_y.value)
        else:
            mat_y_log = mat_y
        with _silence_singular_logm_warning():
            tr_a_log_b = trace_matrix_log(mat_y_log, x_val, m, k, -apx)
            return -ln_quantum_entropy(x_val, m, k) - tr_a_log_b
    elif isinstance(mat_y, cvxpy.Expression) and mat_y.is_constant():
        if mat_y.value is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_y as a numpy.ndarray."
            )
        y_val = np.asarray(mat_y.value)
        if not is_positive_semidefinite(y_val):
            raise ValueError("mat_y must be a positive semidefinite matrix")
        # tr(A log B) with fixed B (MATLAB ``trace(A*logm(B))``).
        if isinstance(mat_x, np.ndarray):
            x_eval = np.asarray(mat_x)
        elif mat_x.value is None:
            raise ValueError("mat_x has no numeric value; pass a numpy.ndarray or set `.value`.")
        else:
            x_eval = np.asarray(mat_x.value)
        with _silence_singular_logm_warning():
            ent_part = -ln_quantum_entropy(mat_x, m, k, -apx)
            log_y = logm(y_val)
            tr_a_log_b = float(np.real(np.trace(x_eval @ log_y)))
            return ent_part - tr_a_log_b

    if not isinstance(mat_x, cvxpy.Expression) or not isinstance(mat_y, cvxpy.Expression):
        raise ValueError("mat_x and mat_y must be numpy arrays or cvxpy expressions")
    if not mat_x.is_affine() or not mat_y.is_affine():
        raise ValueError("mat_x and mat_y must be affine CVXPY expressions.")
    if mat_x.value is None or mat_y.value is None:
        raise ValueError("Affine mat_x and mat_y need numeric initial values; set `.value` for PSD checks.")
    if not is_positive_semidefinite(mat_x.value):
        raise ValueError("mat_x must be positive semidefinite at the initial value.")
    if not is_positive_semidefinite(mat_y.value):
        raise ValueError("mat_y must be positive semidefinite at the initial value.")

    if space_optimized:
        return evaluate_relative_entropy_integral(
            np.asarray(mat_x.value),
            np.asarray(mat_y.value),
            epsilon_dec=epsilon_dec,
            solver=solver,
            **solve_kwargs,
        )

    n = int(mat_x.shape[0])
    is_cplx = np.any(np.imag(mat_x.value) != 0) or np.any(np.imag(mat_y.value) != 0)
    if is_cplx:
        tau = cvxpy.Variable((1, 1), hermitian=True)
    else:
        tau = cvxpy.Variable((1, 1), symmetric=True)

    e = np.reshape(np.eye(n), (-1, 1), order="F")
    mat_x_kron = cvxpy.kron(mat_x, np.eye(n))
    mat_y_kron = cvxpy.kron(np.eye(n), cvxpy.conj(mat_y))
    cons = operator_relative_entropy_epi_cone(
        mat_x_kron,
        mat_y_kron,
        tau,
        m=m,
        k=k,
        e=e,
        apx=apx,
        hermitian=is_cplx,
    )
    obj = tau
    if is_cplx:
        obj = cvxpy.real(obj)
    prob = cvxpy.Problem(cvxpy.Minimize(obj), cons)
    default_kwargs = {"verbose": False}
    default_kwargs.update(solve_kwargs)
    return prob.solve(solver=solver, **default_kwargs)
