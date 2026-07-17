"""CVXPY constraints for the hypograph of the Tsallis entropy."""

import cvxpy

from toqito.cones._utils import _require_square_2d
from toqito.cones.ln_quantum_entropy_hypo_cone import ln_quantum_entropy_hypo_cone
from toqito.cones.trace_matrix_power_hypo_cone import trace_matrix_power_hypo_cone


def tsallis_entropy_hypo_cone(
    mat_x: cvxpy.Expression,
    t: cvxpy.Expression,
    order: float,
    *,
    hermitian: bool = False,
    m: int = 3,
    k: int = 3,
    apx: int = 0,
) -> list[cvxpy.Constraint]:
    r"""Return CVXPY constraints for the hypograph of Tsallis entropy.

    For ``order`` \(r \in (0, 1]\) the map

    \[
        S_r(A) = \frac{1}{r}\operatorname{tr}\!\bigl(A^{1-r} - A\bigr)
    \]

    is concave. The constraints enforce

    \[
        t \leqslant S_r(A)
    \]

    by wrapping ``trace_matrix_power_hypo_cone`` at power \(1-r\): an auxiliary
    scalar ``u`` satisfies

    \[
        u \leqslant \operatorname{tr}(A^{1-r}),
        \qquad
        t \leqslant \frac{u - \operatorname{tr}(A)}{r}.
    \]

    The limit case ``order == 0`` (von Neumann entropy) delegates to
    ``ln_quantum_entropy_hypo_cone`` [@fawzi2015matrixgeometric].

    Args:
        mat_x: A CVXPY expression for a positive semidefinite matrix.
        t: A CVXPY scalar (or ``1 x 1``) hypograph variable.
        order: The Tsallis order \(r \in [0, 1]\) (the float API's ``t``).
        hermitian: Whether the matrices are Hermitian or symmetric.
        m: Quadrature nodes forwarded when ``order == 0``.
        k: Square-root depth forwarded when ``order == 0``.
        apx: Approximation mode forwarded when ``order == 0``.

    Raises:
        ValueError: If ``mat_x`` is not a square 2D expression.
        ValueError: If ``order`` is not in ``[0, 1]``.

    Returns:
        A list of CVXPY constraints.

    Examples:
        ```python exec="1" source="above" result="text"
        import cvxpy
        import numpy as np
        from toqito.cones import tsallis_entropy_hypo_cone

        mat_x = np.diag([0.25, 0.75])
        t = cvxpy.Variable()
        cons = tsallis_entropy_hypo_cone(cvxpy.Constant(mat_x), t, 0.5)
        prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
        print(f"{prob.solve(solver=cvxpy.SCS, verbose=False):.4f}")
        ```

    """
    _require_square_2d(mat_x, "mat_x")
    if order < 0 or order > 1:
        raise ValueError("order must be in the range [0, 1]")

    if order == 0:
        return ln_quantum_entropy_hypo_cone(
            mat_x,
            t,
            m=m,
            k=k,
            apx=apx,
            hermitian=hermitian,
        )

    u = cvxpy.Variable()
    constraints = trace_matrix_power_hypo_cone(
        mat_x,
        u,
        1.0 - order,
        hermitian=hermitian,
    )
    trace_x = cvxpy.trace(mat_x)
    if hermitian:
        trace_x = cvxpy.real(trace_x)
    constraints.append(t <= (u - trace_x) / order)
    return constraints
