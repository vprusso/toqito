r"""Computes \(f(A, B, K, t) = \operatorname{tr}(K^{\dagger} A^{1-t} K B^{t})\) for PSD \(A\) and \(B\)."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones._utils import _contains_effective_variables, _require_2d, _require_square_2d
from toqito.cones.geometric_mean_epi_cone import geometric_mean_epi_cone
from toqito.cones.geometric_mean_hypo_cone import geometric_mean_hypo_cone
from toqito.cones.trace_matrix_power import trace_matrix_power
from toqito.matrix_ops import psd_matrix_power
from toqito.matrix_props import is_positive_semidefinite


def lieb_ando(
    mat_a: np.ndarray | cvxpy.Expression,
    mat_b: np.ndarray | cvxpy.Expression,
    mat_k: np.ndarray,
    t: float,
) -> float:
    r"""Compute \(f(A, B, K, t) = \operatorname{tr}\!\bigl(K^{\dagger} A^{1-t} K B^{t}\bigr)\) for PSD matrices.

    Here \(A=\) ``mat_a``, \(B=\) ``mat_b``, and \(K=\) ``mat_k``. For real data,
    \(K^{\dagger} = K^{\top}\). The map is concave in \((A, B)\) for
    \(t \in [0, 1]\) and convex for \(t \in [-1, 0] \cup [1, 2]\)
    [@fawzi2015matrixgeometric].

    Args:
        mat_a: The first PSD matrix.
        mat_b: The second PSD matrix.
        mat_k: The matrix to multiply the result by.
        t: The power to raise the matrices to. If ``mat_a`` and ``mat_b`` are CVXPY
            expressions, ``t`` must lie in `[-1, 2]`.

    Returns:
        The value of the function as a float.

    Raises:
        TypeError: If ``mat_a`` or ``mat_b`` is not a numpy.ndarray or a cvxpy expression.
        TypeError: If ``mat_k`` is not a numpy.ndarray.
        ValueError: If ``mat_a`` is not 2D or not square.
        ValueError: If ``mat_b`` is not 2D or not square.
        ValueError: If ``mat_k`` is not 2D.
        ValueError: If ``mat_k`` has incompatible shape with ``mat_a`` and ``mat_b``.
        ValueError: If ``mat_a`` and ``mat_b`` are not positive semidefinite.
        ValueError: If ``t`` is not in the range `[-1, 2]` and ``mat_a`` and ``mat_b`` are cvxpy expressions.
        ValueError: If ``mat_a`` and ``mat_b`` are not affine expressions.

    Examples:
        ```python
        import numpy as np
        from toqito.cones import lieb_ando
        mat_a = np.array([[2.0, 1.0], [1.0, 2.0]])
        mat_b = np.array([[2.0, 1.0], [1.0, 2.0]])
        mat_k = np.eye(2)
        t = 0.5
        print(lieb_ando(mat_a, mat_b, mat_k, t))
        ```

    """
    if not isinstance(mat_a, (np.ndarray, cvxpy.Expression)):
        raise TypeError("mat_a must be a numpy.ndarray or a cvxpy expression.")
    if not isinstance(mat_b, (np.ndarray, cvxpy.Expression)):
        raise TypeError("mat_b must be a numpy.ndarray or a cvxpy expression.")
    if not isinstance(mat_k, np.ndarray):
        raise TypeError("mat_k must be a numpy.ndarray.")

    _require_square_2d(mat_a, "mat_a")
    _require_square_2d(mat_b, "mat_b")
    _require_2d(mat_k, "mat_k")
    if mat_k.shape[0] != mat_a.shape[0] or mat_k.shape[1] != mat_b.shape[1]:
        raise ValueError("mat_k must have the same number of rows as mat_a and the same number of columns as mat_b.")

    if isinstance(mat_a, np.ndarray) and isinstance(mat_b, np.ndarray):
        if not is_positive_semidefinite(mat_a) or not is_positive_semidefinite(mat_b):
            raise ValueError("mat_a and mat_b must be positive semidefinite.")
        a_raised = psd_matrix_power(mat_a, 1 - t)
        b_raised = psd_matrix_power(mat_b, t)
        return float(np.real(np.trace(mat_k.conj().T @ a_raised @ mat_k @ b_raised)))
    elif isinstance(mat_a, np.ndarray):
        if not is_positive_semidefinite(mat_a) or not is_positive_semidefinite(mat_b.value):
            raise ValueError("mat_a and mat_b must be positive semidefinite.")
        mat_kak = mat_k.conj().T @ psd_matrix_power(mat_a, 1 - t) @ mat_k
        mat_kak = (mat_kak + mat_kak.conj().T) / 2
        return trace_matrix_power(mat_b, t, mat_kak)
    elif isinstance(mat_b, np.ndarray):
        if not is_positive_semidefinite(mat_a.value) or not is_positive_semidefinite(mat_b):
            raise ValueError("mat_a and mat_b must be positive semidefinite.")
        mat_kkb = mat_k @ psd_matrix_power(mat_b, t) @ mat_k.conj().T
        mat_kkb = (mat_kkb + mat_kkb.conj().T) / 2
        return trace_matrix_power(mat_a, 1 - t, mat_kkb)
    else:
        if _contains_effective_variables(mat_a):
            raise ValueError(
                "mat_a must not contain free CVXPY variables; "
                "use a constant expression or formulate cone constraints directly."
            )
        if _contains_effective_variables(mat_b):
            raise ValueError(
                "mat_b must not contain free CVXPY variables; "
                "use a constant expression or formulate cone constraints directly."
            )
        if not mat_a.is_affine() or not mat_b.is_affine():
            raise ValueError("mat_a and mat_b must be affine expressions.")
        if not is_positive_semidefinite(mat_a.value) or not is_positive_semidefinite(mat_b.value):
            raise ValueError("mat_a and mat_b must be positive semidefinite.")

        n = mat_a.shape[0]
        m = mat_b.shape[0]
        is_cplx = np.any(np.imag(mat_a.value) != 0) or np.any(np.imag(mat_b.value) != 0) or np.any(np.imag(mat_k) != 0)
        Kvec = np.reshape(mat_k.T, (n * m, 1), order="F")
        KvKv = Kvec @ Kvec.conj().T
        KvKv = (KvKv + KvKv.conj().T) / 2
        if is_cplx:
            T = cvxpy.Variable((n * m, n * m), hermitian=True)
        else:
            T = cvxpy.Variable((n * m, n * m), symmetric=True)

        obj = cvxpy.trace(KvKv @ T)
        if is_cplx:
            obj = cvxpy.real(obj)
        mat_a_kron = cvxpy.kron(mat_a, np.eye(m))
        mat_b_kron = cvxpy.kron(np.eye(n), cvxpy.conj(mat_b))
        if t >= 0 and t <= 1:
            cons = geometric_mean_hypo_cone(mat_a_kron, mat_b_kron, T, t, fullhyp=False, hermitian=is_cplx)
            problem = cvxpy.Problem(cvxpy.Maximize(obj), cons)
            result = problem.solve()
            if problem.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
                raise ValueError(f"The SDP did not solve successfully (status: {problem.status}).")
            return result
        elif (t >= -1 and t <= 0) or (t >= 1 and t <= 2):
            cons = geometric_mean_epi_cone(mat_a_kron, mat_b_kron, T, t, hermitian=is_cplx)
            problem = cvxpy.Problem(cvxpy.Minimize(obj), cons)
            result = problem.solve()
            if problem.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
                raise ValueError(f"The SDP did not solve successfully (status: {problem.status}).")
            return result
        else:
            raise ValueError("t must be between -1 and 2.")
