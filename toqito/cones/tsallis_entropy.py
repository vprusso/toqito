"""Tsallis entropy for positive semidefinite matrices."""

import cvxpy
import numpy as np
from scipy.linalg import logm

from toqito.cones._utils import _require_square_2d
from toqito.cones.geometric_mean_hypo_cone import geometric_mean_hypo_cone
from toqito.cones.ln_quantum_entropy import ln_quantum_entropy
from toqito.matrix_ops.psd_matrix_power import psd_matrix_power
from toqito.matrix_props import is_positive_semidefinite


def tsallis_entropy(
    mat_x: np.ndarray | cvxpy.Expression,
    t: float,
) -> float:
    r"""Compute the Tsallis entropy \(S_t(A)\) for a PSD matrix \(A\) [@fawzi2015matrixgeometric].

    For \(t \in [0, 1]\), the Tsallis entropy is defined by

    \[
        S_t(A) = \frac{1}{t}\operatorname{tr}\!\bigl(A^{1-t} - A\bigr).
    \]

    This function uses the natural logarithm implicitly through the limit
    \(S(A) = -\operatorname{tr}(A \log A)\): as \(t \to 0^+\),

    \[
        \lim_{t \to 0^+} S_t(A) = S(A),
    \]

    and \(S_t(A) \geqslant S(A)\) for all \(t \in [0, 1]\). The map \(S_t\) is concave on
    \(\text{H}_n^{++}\) for \(t \in [0, 1]\).

    For affine CVXPY inputs, the hypograph of \(S_t\) is expressed via the matrix geometric
    mean: \(S_t(A) \geqslant \tau\) if and only if there exists \(T\) such that
    \(A \#_t I \succeq T\) and \(\frac{1}{t}\operatorname{tr}(T - A) \geqslant \tau\). The
    constraint \(A \#_t I \succeq T\) is enforced with
    ``geometric_mean_hypo_cone(A, I, T, t, fullhyp=False)``.

    Args:
        mat_x: A positive semidefinite matrix, or an affine CVXPY expression.
        t: Order parameter in the range ``[0, 1]``.

    Raises:
        ValueError: If ``mat_x`` is not a numpy array or a cvxpy expression.
        ValueError: If ``mat_x`` is not a 2D square matrix.
        ValueError: If ``t`` is not in the range ``[0, 1]``.
        ValueError: If ``mat_x`` is not positive semidefinite (at the initial value for
            affine CVXPY inputs).
        ValueError: If ``mat_x`` is not an affine CVXPY expression on the SDP path.
        ValueError: If the SDP does not solve successfully.

    Returns:
        The Tsallis entropy \(S_t(A)\) as a float.

    """
    if not isinstance(mat_x, (np.ndarray, cvxpy.Expression)):
        raise ValueError("mat_x must be a numpy array or a cvxpy expression")
    _require_square_2d(mat_x, "mat_x")
    if t < 0 or t > 1:
        raise ValueError("t must be in the range [0, 1]")

    if isinstance(mat_x, np.ndarray):
        if not is_positive_semidefinite(mat_x):
            raise ValueError("mat_x must be a positive semidefinite matrix")
        if t == 0:
            return float(np.real(-np.trace(mat_x @ logm(mat_x))))
        mat_power = psd_matrix_power(mat_x, 1 - t)
        return float(np.real((np.trace(mat_power) - np.trace(mat_x)) / t))

    if isinstance(mat_x, cvxpy.Expression) and mat_x.is_constant():
        x_val = mat_x.value
        if x_val is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_x as a numpy.ndarray."
            )
        return tsallis_entropy(np.asarray(x_val), t)

    if not mat_x.is_affine():
        raise ValueError("mat_x must be an affine CVXPY expression.")
    if mat_x.value is None:
        raise ValueError(
            "Affine mat_x has no numeric initial value; set `.value` for PSD checks."
        )
    if not is_positive_semidefinite(mat_x.value):
        raise ValueError("mat_x must be positive semidefinite at the initial value.")

    if t == 0:
        return ln_quantum_entropy(mat_x)

    n = int(mat_x.shape[0])
    is_cplx = np.any(np.imag(mat_x.value) != 0)
    if is_cplx:
        mat_t = cvxpy.Variable((n, n), hermitian=True)
    else:
        mat_t = cvxpy.Variable((n, n), symmetric=True)
    tau = cvxpy.Variable()
    eye_n = cvxpy.Constant(np.eye(n))

    constraints = geometric_mean_hypo_cone(
        mat_x,
        eye_n,
        mat_t,
        t,
        fullhyp=False,
        hermitian=is_cplx,
    )
    trace_diff = cvxpy.trace(mat_t - mat_x)
    if is_cplx:
        trace_diff = cvxpy.real(trace_diff)
    constraints.append((1 / t) * trace_diff >= tau)

    prob = cvxpy.Problem(cvxpy.Maximize(tau), constraints)
    result = prob.solve(solver=cvxpy.SCS, verbose=False)
    if prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
        raise ValueError(f"The SDP did not solve successfully (status: {prob.status}).")
    return float(result)
