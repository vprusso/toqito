r"""Quantum conditional entropy for bipartite positive semidefinite matrices."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones._utils import _reject_nonconstant_cvxpy
from toqito.matrix_ops.partial_trace import partial_trace
from toqito.state_props.quantum_relative_entropy import quantum_relative_entropy


def quantum_conditional_entropy(
    rho: np.ndarray | cvxpy.Expression,
    dim: list[int] | np.ndarray,
    sys: int = 0,
) -> float:
    r"""Compute the quantum conditional entropy \(H(A|B)\) or \(H(B|A)\) for a bipartite state.

    The quantum conditional entropy is defined as

    \[
        H(A|B) = -D(\rho_{AB}\parallel I_A \otimes \rho_B)
    \]

    or

    \[
        H(B|A) = -D(\rho_{AB}\parallel \rho_A \otimes I_B),
    \]

    where \(D(\cdot\parallel\cdot)\) is the quantum relative entropy in nats.

    Note that this function uses the natural logarithm (base e), consistent with
    `toqito.state_props.quantum_relative_entropy`.

    This function evaluates the formula numerically. Constant CVXPY expressions
    with a concrete ``.value`` are routed through the numeric path. Affine or
    variable CVXPY inputs are not supported; compose
    ``quantum_relative_entropy_epi_cone`` with the appropriate Kronecker lift
    in a parent SDP.

    Args:
        rho: Bipartite positive semidefinite matrix (or constant CVXPY expression).
        dim: Subsystem dimensions ``[n_a, n_b]``.
        sys: ``0`` for \(H(A|B)\), ``1`` for \(H(B|A)\) (0-indexed partial trace).

    Raises:
        ValueError: If ``sys`` is not ``0`` or ``1``.
        ValueError: If ``rho`` is not a numpy array or a cvxpy expression.
        ValueError: If ``dim`` is not a list or numpy array.
        ValueError: If ``dim`` does not have length 2.
        ValueError: If ``dim`` does not have integer elements.
        ValueError: If ``dim`` does not have positive elements.
        ValueError: If ``dim`` does not match the shape of ``rho``.
        ValueError: If a constant CVXPY expression has no numeric ``.value``.
        ValueError: If affine or variable CVXPY inputs are passed.

    Returns:
        The conditional entropy in nats.

    Examples:
        Compute \(H(A|B)\) for a Bell state and for a product state:

        ```python exec="1" source="above" result="text"
        import numpy as np

        from toqito.state_props import quantum_conditional_entropy

        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        bell_state = np.outer(psi, psi.conj())
        product_state = np.diag([1 / 2, 0, 1 / 2, 0])

        bell_entropy = quantum_conditional_entropy(bell_state, [2, 2], sys=0)
        product_entropy = quantum_conditional_entropy(product_state, [2, 2], sys=0)

        print(f"Bell state H(A|B): {bell_entropy:.6f}")
        print(f"Product state H(A|B): {product_entropy:.6f}")
        ```

    """
    if sys not in [0, 1]:
        raise ValueError("sys must be 0 or 1")
    if not isinstance(rho, (np.ndarray, cvxpy.Expression)):
        raise ValueError("rho must be a numpy array or a cvxpy expression")
    if not isinstance(dim, (list, np.ndarray)):
        raise ValueError("dim must be a list or numpy array")
    if len(dim) != 2:
        raise ValueError("dim must have length 2")
    if not isinstance(dim[0], int) or not isinstance(dim[1], int):
        raise ValueError("dim must have integer elements")
    if dim[0] <= 0 or dim[1] <= 0:
        raise ValueError("dim must have positive elements")
    if dim[0] * dim[1] != rho.shape[0]:
        raise ValueError("dim must match the shape of rho")

    if isinstance(rho, cvxpy.Expression) and rho.is_constant():
        rho_val = rho.value
        if rho_val is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` or pass rho as a numpy.ndarray."
            )
        return quantum_conditional_entropy(np.asarray(rho_val), dim, sys)

    if isinstance(rho, np.ndarray):
        if sys == 0:
            sigma = np.kron(np.eye(dim[0]), partial_trace(rho, 0, dim))
        else:
            sigma = np.kron(partial_trace(rho, 1, dim), np.eye(dim[1]))
        return -quantum_relative_entropy(rho, sigma)

    _reject_nonconstant_cvxpy(rho)
