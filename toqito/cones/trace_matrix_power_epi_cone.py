"""CVXPY constraints for the epigraph of the matrix-power trace."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones._utils import _is_psd_matrix, _require_square_2d, _symmetric_like_variable
from toqito.cones.geometric_mean_epi_cone import geometric_mean_epi_cone


def trace_matrix_power_epi_cone(
    mat_a: cvxpy.Expression,
    t: cvxpy.Expression,
    power: float,
    mat_c: np.ndarray | None = None,
    *,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Return CVXPY constraints for the epigraph of \(\operatorname{tr}(C A^{p})\).

    For ``power`` \(p \in [-1, 0] \cup [1, 2]\) the map
    \(A \mapsto \operatorname{tr}(C A^{p})\) is convex in \(A\) (with fixed PSD
    weight \(C\)). The constraints enforce

    \[
        t \geqslant \operatorname{tr}(C A^{p})
    \]

    approximately via the matrix geometric mean identity
    \(A^{p} = G_{p}(I, A)\). An auxiliary matrix ``T`` is introduced internally with

    \[
        G_{p}(I, A) \preceq T,
        \qquad
        t \geqslant \operatorname{tr}(C\, T)
    \]

    as in CVXQUAD ``trace_mpow.m`` [@fawzi2015matrixgeometric].

    If ``mat_c`` is omitted, it defaults to the identity.

    Args:
        mat_a: A CVXPY expression for a positive semidefinite matrix.
        t: A CVXPY scalar (or ``1 x 1``) epigraph variable.
        power: The matrix power \(p \in [-1, 0] \cup [1, 2]\).
        mat_c: A fixed positive semidefinite weight matrix (default: identity).
        hermitian: Whether the matrices are Hermitian or symmetric.

    Raises:
        ValueError: If ``mat_a`` is not a square 2D expression.
        ValueError: If ``mat_c`` is not a square PSD numpy array of matching shape.
        ValueError: If ``power`` is not in ``[-1, 0]`` or ``[1, 2]``.

    Returns:
        A list of CVXPY constraints.

    Examples:
        ```python exec="1" source="above" result="text"
        import cvxpy
        import numpy as np
        from toqito.cones import trace_matrix_power_epi_cone

        n = 2
        mat_a = np.array([[2.0, 1.0], [1.0, 2.0]])
        a_c = cvxpy.Constant(mat_a)
        t = cvxpy.Variable()
        cons = trace_matrix_power_epi_cone(a_c, t, 1.5)
        prob = cvxpy.Problem(cvxpy.Minimize(t), cons)
        print(f"{prob.solve(solver=cvxpy.SCS, verbose=False):.4f}")
        ```

    """
    _require_square_2d(mat_a, "mat_a")
    if power < -1 or (power > 0 and power < 1) or power > 2:
        raise ValueError("power must be in the range [-1, 0] or [1, 2]")
    n = int(mat_a.shape[0])
    if mat_c is None:
        mat_c = np.eye(n)
    else:
        if not isinstance(mat_c, np.ndarray):
            raise ValueError("mat_c must be a numpy array")
        _require_square_2d(mat_c, "mat_c")
        if not _is_psd_matrix(mat_c):
            raise ValueError("mat_c must be a positive semidefinite matrix")
    if mat_a.shape != mat_c.shape:
        raise ValueError("mat_a and mat_c must have the same shape")

    tau = _symmetric_like_variable(n, hermitian=hermitian)
    eye_n = cvxpy.Constant(np.eye(n))
    constraints = geometric_mean_epi_cone(
        eye_n,
        mat_a,
        tau,
        power,
        hermitian=hermitian,
    )
    weighted_trace = cvxpy.trace(cvxpy.Constant(mat_c) @ tau)
    if hermitian:
        weighted_trace = cvxpy.real(weighted_trace)
    constraints.append(t >= weighted_trace)
    return constraints
