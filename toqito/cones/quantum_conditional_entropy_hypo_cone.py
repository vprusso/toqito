"""CVXPY constraints for the hypograph of quantum conditional entropy."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import numbers

import cvxpy
import numpy as np

from toqito.cones._utils import _require_square_2d
from toqito.cones.quantum_relative_entropy_epi_cone import (
    quantum_relative_entropy_epi_cone,
)


def quantum_conditional_entropy_hypo_cone(
    rho: cvxpy.Expression,
    t: cvxpy.Expression,
    dim: list[int] | np.ndarray,
    sys: int = 0,
    *,
    m: int = 3,
    k: int = 3,
    apx: int = 0,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Return CVXPY constraints for the hypograph of quantum conditional entropy.

    For a bipartite PSD state \(\rho_{AB}\) with dimensions ``dim = [n_a, n_b]``,

    \[
        H(A|B) = -D(\rho_{AB}\parallel I_A \otimes \rho_B),
        \qquad
        H(B|A) = -D(\rho_{AB}\parallel \rho_A \otimes I_B),
    \]

    where \(D\) is quantum relative entropy (nats). The map \(\rho \mapsto H(\cdot|\cdot)\)
    is concave. The constraints enforce

    \[
        t \leqslant H(A|B)
        \quad\text{or}\quad
        t \leqslant H(B|A)
    \]

    according to ``sys``, by wrapping ``quantum_relative_entropy_epi_cone``:
    an auxiliary scalar ``tau`` satisfies

    \[
        \tau \geqslant D(\rho\parallel\sigma),
        \qquad
        t \leqslant -\tau,
    \]

    with \(\sigma = I_A \otimes \rho_B\) when ``sys == 0`` and
    \(\sigma = \rho_A \otimes I_B\) when ``sys == 1``
    [@fawzi2017matrixlogarithm].

    Args:
        rho: A CVXPY expression for an ``(n_a n_b) x (n_a n_b)`` PSD matrix.
        t: A CVXPY scalar (or ``1 x 1``) hypograph variable.
        dim: Subsystem dimensions ``[n_a, n_b]``.
        sys: ``0`` for \(H(A|B)\), ``1`` for \(H(B|A)\).
        m: Quadrature nodes forwarded to ``quantum_relative_entropy_epi_cone``.
        k: Square-root depth forwarded to ``quantum_relative_entropy_epi_cone``.
        apx: Approximation mode forwarded to ``quantum_relative_entropy_epi_cone``.
        hermitian: Whether the matrices are Hermitian or symmetric.

    Raises:
        ValueError: If ``sys`` is not ``0`` or ``1``.
        ValueError: If ``rho`` is not a square 2D expression.
        ValueError: If ``dim`` is invalid or does not match ``rho``.

    Returns:
        A list of CVXPY constraints.

    Examples:
        ```python exec="1" source="above" result="text"
        import cvxpy
        import numpy as np
        from toqito.cones import quantum_conditional_entropy_hypo_cone

        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        bell = np.outer(psi, psi.conj())
        t = cvxpy.Variable()
        cons = quantum_conditional_entropy_hypo_cone(
            cvxpy.Constant(bell), t, [2, 2], sys=0
        )
        prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
        print(f"{prob.solve(solver=cvxpy.SCS, verbose=False):.4f}")
        ```

    """
    if sys not in (0, 1):
        raise ValueError("sys must be 0 or 1")
    _require_square_2d(rho, "rho")
    if not isinstance(dim, (list, np.ndarray)):
        raise ValueError("dim must be a list or numpy array")
    if len(dim) != 2:
        raise ValueError("dim must have length 2")
    if not isinstance(dim[0], numbers.Integral) or not isinstance(dim[1], numbers.Integral):
        raise ValueError("dim must have integer elements")
    if dim[0] <= 0 or dim[1] <= 0:
        raise ValueError("dim must have positive elements")
    if int(dim[0]) * int(dim[1]) != int(rho.shape[0]):
        raise ValueError("dim must match the shape of rho")

    dim_list = [int(dim[0]), int(dim[1])]
    dims = (dim_list[0], dim_list[1])
    if sys == 0:
        sigma = cvxpy.kron(np.eye(dim_list[0]), cvxpy.partial_trace(rho, dims, axis=0))
    else:
        sigma = cvxpy.kron(cvxpy.partial_trace(rho, dims, axis=1), np.eye(dim_list[1]))

    tau = cvxpy.Variable()
    constraints = quantum_relative_entropy_epi_cone(
        rho,
        sigma,
        tau,
        m=m,
        k=k,
        apx=apx,
        hermitian=hermitian,
    )
    bound = tau
    if hermitian:
        bound = cvxpy.real(bound)
    constraints.append(t <= -bound)
    return constraints
