"""CVXPY constraints for the hypograph of the Lieb--Ando function."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones._utils import _require_2d, _require_square_2d, _symmetric_like_variable
from toqito.cones.geometric_mean_hypo_cone import geometric_mean_hypo_cone


def lieb_ando_hypo_cone(
    mat_a: cvxpy.Expression,
    mat_b: cvxpy.Expression,
    mat_k: np.ndarray,
    t: cvxpy.Expression,
    power: float,
    *,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Return CVXPY constraints for the hypograph of Lieb's function.

    For ``power`` \(p \in [0, 1]\) the map
    \((A, B) \mapsto \operatorname{tr}(K^{\dagger} A^{1-p} K B^{p})\)
    is jointly concave. The constraints enforce

    \[
        t \leqslant \operatorname{tr}\!\bigl(K^{\dagger} A^{1-p} K B^{p}\bigr)
    \]

    via the Kronecker lifting used in CVXQUAD ``lieb_ando.m``
    [@fawzi2015matrixgeometric]: an auxiliary matrix ``T`` satisfies

    \[
        G_{p}\!\bigl(A \otimes I,\; I \otimes \overline{B}\bigr) \succeq T,
        \qquad
        t \leqslant \operatorname{tr}(\mathrm{vec}(K^{\top})\,\mathrm{vec}(K^{\top})^{H}\, T).
    \]

    Args:
        mat_a: A CVXPY expression for an ``n x n`` PSD matrix.
        mat_b: A CVXPY expression for an ``m x m`` PSD matrix.
        mat_k: A fixed numpy weight of shape ``(n, m)``.
        t: A CVXPY scalar (or ``1 x 1``) hypograph variable.
        power: The Lieb--Ando exponent \(p \in [0, 1]\).
        hermitian: Whether the matrices are Hermitian or symmetric.

    Raises:
        ValueError: If ``mat_a`` or ``mat_b`` is not square 2D.
        ValueError: If ``mat_k`` is not 2D or has incompatible shape.
        ValueError: If ``mat_k`` is not a numpy array.
        ValueError: If ``power`` is not in ``[0, 1]``.

    Returns:
        A list of CVXPY constraints.

    Examples:
        ```python exec="1" source="above" result="text"
        import cvxpy
        import numpy as np
        from toqito.cones import lieb_ando_hypo_cone

        mat_a = np.array([[2.0, 1.0], [1.0, 2.0]])
        mat_b = np.array([[2.0, 1.0], [1.0, 2.0]])
        mat_k = np.eye(2)
        t = cvxpy.Variable()
        cons = lieb_ando_hypo_cone(
            cvxpy.Constant(mat_a),
            cvxpy.Constant(mat_b),
            mat_k,
            t,
            0.5,
        )
        prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
        print(f"{prob.solve(solver=cvxpy.SCS, verbose=False):.4f}")
        ```

    """
    _require_square_2d(mat_a, "mat_a")
    _require_square_2d(mat_b, "mat_b")
    if not isinstance(mat_k, np.ndarray):
        raise ValueError("mat_k must be a numpy array")
    _require_2d(mat_k, "mat_k")
    if mat_k.shape[0] != mat_a.shape[0] or mat_k.shape[1] != mat_b.shape[1]:
        raise ValueError(
            "mat_k must have the same number of rows as mat_a and the same number of columns as mat_b."
        )
    if power < 0 or power > 1:
        raise ValueError("power must be in the range [0, 1]")

    n = int(mat_a.shape[0])
    m = int(mat_b.shape[0])
    k_vec = np.reshape(mat_k.T, (n * m, 1), order="F")
    kvkv = k_vec @ k_vec.conj().T
    kvkv = (kvkv + kvkv.conj().T) / 2

    tau = _symmetric_like_variable(n * m, hermitian=hermitian)
    mat_a_kron = cvxpy.kron(mat_a, np.eye(m))
    mat_b_kron = cvxpy.kron(np.eye(n), cvxpy.conj(mat_b))
    constraints = geometric_mean_hypo_cone(
        mat_a_kron,
        mat_b_kron,
        tau,
        power,
        fullhyp=False,
        hermitian=hermitian,
    )
    weighted_trace = cvxpy.trace(cvxpy.Constant(kvkv) @ tau)
    if hermitian:
        weighted_trace = cvxpy.real(weighted_trace)
    constraints.append(t <= weighted_trace)
    return constraints
