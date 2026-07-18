"""CVXPY constraints for the epigraph of Tsallis relative entropy."""

import cvxpy
import numpy as np

from toqito.cones._utils import _require_square_2d
from toqito.cones.lieb_ando_hypo_cone import lieb_ando_hypo_cone
from toqito.cones.quantum_relative_entropy_epi_cone import (
    quantum_relative_entropy_epi_cone,
)


def tsallis_relative_entropy_epi_cone(
    mat_x: cvxpy.Expression,
    mat_y: cvxpy.Expression,
    t: cvxpy.Expression,
    order: float,
    *,
    hermitian: bool = False,
    m: int = 3,
    k: int = 3,
    apx: int = 0,
) -> list[cvxpy.Constraint]:
    r"""Return CVXPY constraints for the epigraph of Tsallis relative entropy.

    For ``order`` \(r \in (0, 1]\) the map

    \[
        S_r(A\|B) = \frac{1}{r}\operatorname{tr}\!\bigl(A - A^{1-r} B^{r}\bigr)
    \]

    is jointly convex. The constraints enforce

    \[
        t \geqslant S_r(A\|B)
    \]

    by wrapping ``lieb_ando_hypo_cone`` at power \(r\) with identity weight: an
    auxiliary scalar ``u`` satisfies

    \[
        u \leqslant \operatorname{tr}\!\bigl(A^{1-r} B^{r}\bigr),
        \qquad
        t \geqslant \frac{\operatorname{tr}(A) - u}{r}.
    \]

    The limit case ``order == 0`` (quantum relative entropy) delegates to
    ``quantum_relative_entropy_epi_cone`` [@fawzi2015matrixgeometric].

    Args:
        mat_x: A CVXPY expression for an ``n x n`` PSD matrix \(A\).
        mat_y: A CVXPY expression for an ``n x n`` PSD matrix \(B\).
        t: A CVXPY scalar (or ``1 x 1``) epigraph variable.
        order: The Tsallis order \(r \in [0, 1]\) (the float API's ``t``).
        hermitian: Whether the matrices are Hermitian or symmetric.
        m: Quadrature nodes forwarded when ``order == 0``.
        k: Square-root depth forwarded when ``order == 0``.
        apx: Approximation mode forwarded when ``order == 0``.

    Raises:
        ValueError: If ``mat_x`` or ``mat_y`` is not a square 2D expression.
        ValueError: If ``mat_x`` and ``mat_y`` do not have the same shape.
        ValueError: If ``order`` is not in ``[0, 1]``.

    Returns:
        A list of CVXPY constraints.

    Examples:
        ```python exec="1" source="above" result="text"
        import cvxpy
        import numpy as np
        from toqito.cones import tsallis_relative_entropy_epi_cone

        mat_x = np.diag([0.25, 0.75])
        mat_y = np.diag([0.5, 0.5])
        t = cvxpy.Variable()
        cons = tsallis_relative_entropy_epi_cone(
            cvxpy.Constant(mat_x),
            cvxpy.Constant(mat_y),
            t,
            0.5,
        )
        prob = cvxpy.Problem(cvxpy.Minimize(t), cons)
        print(f"{prob.solve(solver=cvxpy.SCS, verbose=False):.4f}")
        ```

    """
    _require_square_2d(mat_x, "mat_x")
    _require_square_2d(mat_y, "mat_y")
    if mat_x.shape != mat_y.shape:
        raise ValueError("mat_x and mat_y must have the same shape")
    if order < 0 or order > 1:
        raise ValueError("order must be in the range [0, 1]")

    if order == 0:
        return quantum_relative_entropy_epi_cone(
            mat_x,
            mat_y,
            t,
            m=m,
            k=k,
            apx=apx,
            hermitian=hermitian,
        )

    n = int(mat_x.shape[0])
    u = cvxpy.Variable()
    constraints = lieb_ando_hypo_cone(
        mat_x,
        mat_y,
        np.eye(n),
        u,
        order,
        hermitian=hermitian,
    )
    trace_x = cvxpy.trace(mat_x)
    if hermitian:
        trace_x = cvxpy.real(trace_x)
    constraints.append(t >= (trace_x - u) / order)
    return constraints
