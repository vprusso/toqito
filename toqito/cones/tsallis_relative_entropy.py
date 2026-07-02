r"""Tsallis relative entropy for positive semidefinite matrices."""

import cvxpy
import numpy as np

from toqito.cones._utils import _require_square_2d
from toqito.cones.geometric_mean_hypo_cone import geometric_mean_hypo_cone
from toqito.cones.lieb_ando import lieb_ando
from toqito.matrix_props import is_positive_semidefinite


def tsallis_relative_entropy(
    mat_x: np.ndarray | cvxpy.Expression,
    mat_y: np.ndarray | cvxpy.Expression,
    t: float,
) -> float:
    r"""Compute the Tsallis relative entropy \(S_t(A\|B)\) [@fawzi2015matrixgeometric].

    For \(t \in [0, 1]\), the Tsallis relative entropy is defined by

    \[
        S_t(A\|B) = \frac{1}{t}\operatorname{tr}\!\bigl(A - A^{1-t} B^t\bigr).
    \]

    As \(t \to 0^+\), this converges to the quantum relative entropy

    \[
        D(A\|B) = \operatorname{tr}\!\bigl[A(\log A - \log B)\bigr],
    \]

    with convergence from below, i.e. \(S_t(A\|B) \leqslant D(A\|B)\) for all
    \(t \in [0, 1]\). The map \((A, B) \mapsto S_t(A\|B)\) is jointly convex on
    \(\text{H}_n^{++} \times \text{H}_n^{++}\).

    For affine CVXPY inputs, the epigraph of \(S_t(\cdot\|\cdot)\) admits the
    description: \(S_t(A\|B) \leqslant \tau\) if and only if there exists a scalar
    \(s\) such that

    \[
        \operatorname{tr}\!\bigl(A^{1-t} B^t\bigr) \geqslant s,
        \qquad
        \frac{1}{t}\bigl[\operatorname{tr}(A) - s\bigr] \leqslant \tau.
    \]

    The constraint \(\operatorname{tr}(A^{1-t} B^t) \geqslant s\) is enforced via
    ``lieb_ando(A, B, I, t)`` with \(K = I\).

    Args:
        mat_x: The first positive semidefinite matrix \(A\), or an affine CVXPY expression.
        mat_y: The second positive semidefinite matrix \(B\), or an affine CVXPY expression.
        t: Order parameter in the range ``[0, 1]``.

    Raises:
        ValueError: If ``mat_x`` or ``mat_y`` is not a numpy array or a cvxpy expression.
        ValueError: If ``mat_x`` or ``mat_y`` is not a 2D square matrix.
        ValueError: If ``mat_x`` and ``mat_y`` do not have the same shape.
        ValueError: If ``t`` is not in the range ``[0, 1]``.
        ValueError: If ``mat_x`` or ``mat_y`` is not positive semidefinite (at the initial
            value for affine CVXPY inputs).
        ValueError: If ``mat_x`` or ``mat_y`` is not an affine CVXPY expression on the
            SDP path.
        ValueError: If the SDP does not solve successfully.

    Returns:
        The Tsallis relative entropy \(S_t(A\|B)\) as a float.

    """
    if not isinstance(mat_x, (np.ndarray, cvxpy.Expression)):
        raise ValueError("mat_x must be a numpy array or a cvxpy expression")
    if not isinstance(mat_y, (np.ndarray, cvxpy.Expression)):
        raise ValueError("mat_y must be a numpy array or a cvxpy expression")
    _require_square_2d(mat_x, "mat_x")
    _require_square_2d(mat_y, "mat_y")
    if mat_x.shape != mat_y.shape:
        raise ValueError("mat_x and mat_y must have the same shape")
    if t < 0 or t > 1:
        raise ValueError("t must be in the range [0, 1]")

    if isinstance(mat_x, np.ndarray) and isinstance(mat_y, np.ndarray):
        if not is_positive_semidefinite(mat_x):
            raise ValueError("mat_x must be a positive semidefinite matrix")
        if not is_positive_semidefinite(mat_y):
            raise ValueError("mat_y must be a positive semidefinite matrix")
        if t == 0:
            from toqito.state_props.quantum_relative_entropy import (
                quantum_relative_entropy,
            )

            return quantum_relative_entropy(mat_x, mat_y)
        n = int(mat_x.shape[0])
        trace_cross = lieb_ando(mat_x, mat_y, np.eye(n), t)
        return float(np.real((np.trace(mat_x) - trace_cross) / t))

    if isinstance(mat_x, np.ndarray):
        mat_x = cvxpy.Constant(mat_x)
    if isinstance(mat_y, np.ndarray):
        mat_y = cvxpy.Constant(mat_y)

    if mat_x.is_constant() and mat_y.is_constant():
        if mat_x.value is None or mat_y.value is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass a numpy.ndarray."
            )
        return tsallis_relative_entropy(
            np.asarray(mat_x.value), np.asarray(mat_y.value), t
        )

    if not mat_x.is_affine() or not mat_y.is_affine():
        raise ValueError("mat_x and mat_y must be affine CVXPY expressions.")
    if mat_x.value is None or mat_y.value is None:
        raise ValueError(
            "Affine mat_x and mat_y need numeric initial values; set `.value` for PSD checks."
        )
    if not is_positive_semidefinite(mat_x.value):
        raise ValueError("mat_x must be positive semidefinite at the initial value.")
    if not is_positive_semidefinite(mat_y.value):
        raise ValueError("mat_y must be positive semidefinite at the initial value.")

    if t == 0:
        from toqito.state_props.quantum_relative_entropy import quantum_relative_entropy

        return quantum_relative_entropy(mat_x, mat_y)

    n = int(mat_x.shape[0])
    is_cplx = np.any(np.imag(mat_x.value) != 0) or np.any(np.imag(mat_y.value) != 0)
    s_var = cvxpy.Variable()
    tau_var = cvxpy.Variable()
    eye_n = np.eye(n)
    k_vec = np.reshape(eye_n.T, (n * n, 1), order="F")
    k_vk = k_vec @ k_vec.conj().T
    k_vk = (k_vk + k_vk.conj().T) / 2
    if is_cplx:
        mat_t = cvxpy.Variable((n * n, n * n), hermitian=True)
    else:
        mat_t = cvxpy.Variable((n * n, n * n), symmetric=True)

    mat_x_kron = cvxpy.kron(mat_x, np.eye(n))
    mat_y_kron = cvxpy.kron(np.eye(n), cvxpy.conj(mat_y))
    constraints = geometric_mean_hypo_cone(
        mat_x_kron,
        mat_y_kron,
        mat_t,
        t,
        fullhyp=False,
        hermitian=is_cplx,
    )
    lieb_expr = cvxpy.trace(k_vk @ mat_t)
    if is_cplx:
        lieb_expr = cvxpy.real(lieb_expr)
    constraints.append(lieb_expr >= s_var)

    trace_a = cvxpy.trace(mat_x)
    if is_cplx:
        trace_a = cvxpy.real(trace_a)
    constraints.append(tau_var >= (1 / t) * (trace_a - s_var))

    prob = cvxpy.Problem(cvxpy.Minimize(tau_var), constraints)
    result = prob.solve(solver=cvxpy.SCS, verbose=False)
    if prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
        raise ValueError(f"The SDP did not solve successfully (status: {prob.status}).")
    return float(result)
