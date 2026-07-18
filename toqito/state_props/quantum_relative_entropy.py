r"""Quantum relative entropy for positive semidefinite matrices."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import contextlib
import warnings

import cvxpy
import numpy as np

from toqito.cones._utils import _reject_nonconstant_cvxpy, _require_square_2d
from toqito.matrix_props import is_hermitian, is_positive_semidefinite


@contextlib.contextmanager
def _silence_singular_logm_warning():
    """Silence scipy's cosmetic "logm input matrix is exactly singular" warning.

    The numeric relative-entropy path evaluates eigendecompositions on
    rank-deficient density matrices, which is routine for pure states and for
    reduced/lifted operators used by callers such as
    ``quantum_conditional_entropy``. The singular directions contribute the
    correct ``0 * (-inf) = 0`` to the trace, so the returned value is valid and
    the warning is noise. The filter matches only that exact message, so genuine
    warnings still propagate.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The logm input matrix is exactly singular.")
        yield


def quantum_relative_entropy(
    mat_x: np.ndarray | cvxpy.Expression,
    mat_y: np.ndarray | cvxpy.Expression,
) -> float:
    r"""Compute the quantum relative entropy \(D(X||Y)\) for PSD \(X\) and \(Y\).

    Note that this function uses the natural logarithm (base e) for the
    entropy calculation and not the base 2 logarithm.

    The quantum relative entropy is jointly convex in \(X\) and \(Y\), assuming
    that \(X\) and \(Y\) are positive semidefinite matrices.

    This function evaluates the formula numerically. Constant CVXPY expressions
    with a concrete ``.value`` are routed through the numeric path. Affine or
    variable CVXPY inputs are not supported; use
    ``quantum_relative_entropy_epi_cone`` for composition in a parent SDP.
    For the space-optimized integral SDP representation, use
    ``integral_relative_entropy_lower_cone`` /
    ``integral_relative_entropy_upper_cone``, or call
    ``evaluate_relative_entropy_integral`` for numeric bounds
    [@kossmann2024optimisingrelativeentropy].

    Args:
        mat_x: A numpy array or constant CVXPY expression for the first PSD matrix.
        mat_y: A numpy array or constant CVXPY expression for the second PSD matrix.

    Raises:
        ValueError: If ``mat_x`` or ``mat_y`` is not a numpy array or cvxpy expression.
        ValueError: If the input matrices are not square or 2D.
        ValueError: If the input matrices are not positive semidefinite.
        ValueError: If the input matrices are not Hermitian.
        ValueError: If ``mat_x`` and ``mat_y`` do not have the same shape.
        ValueError: If a constant CVXPY expression has no numeric ``.value``.
        ValueError: If affine or variable CVXPY inputs are passed.
        ValueError: If \(D(X\|Y)\) is infinite because \(\mathrm{im}(X)\) is not
            contained in \(\mathrm{im}(Y)\).

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

    if isinstance(mat_x, cvxpy.Expression) and mat_x.is_constant():
        x_val = mat_x.value
        if x_val is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_x as a numpy.ndarray."
            )
        return quantum_relative_entropy(np.asarray(x_val), mat_y)

    if isinstance(mat_y, cvxpy.Expression) and mat_y.is_constant():
        y_val = mat_y.value
        if y_val is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_y as a numpy.ndarray."
            )
        return quantum_relative_entropy(mat_x, np.asarray(y_val))

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
        with _silence_singular_logm_warning():
            eig_x, vec_x = np.linalg.eigh(mat_x)
            eig_y, vec_y = np.linalg.eigh(mat_y)
            overlaps = vec_x.conj().T @ vec_y
            u = eig_x @ (np.abs(overlaps) ** 2)

            if np.any(u[eig_y <= tol] > tol):
                raise ValueError("D(X||Y) is infinity because im(X) is not contained in im(Y)")
            r1 = np.sum(eig_x[eig_x > tol] * np.log(eig_x[eig_x > tol]))
            r2 = np.sum(u[eig_y > tol] * np.log(eig_y[eig_y > tol]))
            return float(r1 - r2)

    _reject_nonconstant_cvxpy(mat_x, mat_y)
