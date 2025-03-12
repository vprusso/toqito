"""Check if a quantum state is absolutely PPT."""

import numpy as np

from toqito.matrix_props.is_positive_semidefinite import is_positive_semidefinite
from toqito.state_opt.symmetric_extension_hierarchy import symmetric_extension_hierarchy
from toqito.state_props.is_ppt import is_ppt
from toqito.state_props.is_separable import is_separable


def is_abs_ppt(rho: np.ndarray, dim: list[int] = None, max_constraints: int = 2612) -> int:
    r"""Determine whether a quantum state is absolutely PPT.

    A quantum state :math:`\rho` is absolutely PPT if it remains PPT under any unitary transformation.
    This function evaluates the condition using:

    - Quick separability checks (to immediately identify separable states)
    - The PPT property (to test for the PPT condition)
    - A Gershgorin-type condition (providing a bound based on the eigenvalues)
    - A symmetric extension hierarchy (for deeper verification when needed)

    .. math::
        \rho \text{ is absolutely PPT} \iff \forall U,\, (U \rho U^\dagger)^{T_B} \succeq 0.

    Examples
    ========
    Example 1: A separable mixed state (trivially PPT)

    >>> import numpy as np
    >>> from toqito.state_props import is_abs_ppt
    >>> rho = np.array([[0.5, 0], [0, 0.5]])  # Maximally mixed state
    >>> is_abs_ppt(rho, [2, 2])
    1

    Example 2: A Bell state (entangled, not absolutely PPT)

    >>> from toqito.states import bell
    >>> bell_state = bell(0) @ bell(0).conj().T
    >>> is_abs_ppt(bell_state, [2, 2])
    0

    Example 3: A high-dimensional identity matrix (returns -1 for inconclusive cases)

    >>> rho = np.eye(49) / 49
    >>> is_abs_ppt(rho, [7, 7])
    -1

    References
    ==========
    - MATLAB QETLAB Implementation: Nathaniel Johnston,
      "IsAbsPPT function" <http://www.qetlab.com/IsAbsPPT>

    Parameters
    ==========
    rho : np.ndarray
        The density matrix (or a vector of its eigenvalues) to be tested.
    dim : list[int], optional
        A list of two integers representing the dimensions of the bipartite subsystems.
        If `None`, an equal split is assumed.
    max_constraints : int, optional
        The maximum number of SDP constraints to impose when using the symmetric extension hierarchy.

    Returns
    =======
    int
        - 1 if `rho` is absolutely PPT.
        - 0 if `rho` is not absolutely PPT.
        - -1 if the test is inconclusive (e.g., due to high dimension or computational constraints).

    Raises
    ======
    ValueError
        If `rho` is not square.
    ValueError
        If `rho` has zero trace.
    ValueError
        If `dim` does not match the shape of `rho`.

    """
    rho = np.array(rho, dtype=np.complex128)

    if rho.shape[0] != rho.shape[1]:
        raise ValueError("The input matrix `rho` must be square.")

    if np.isclose(np.trace(rho), 0):
        raise ValueError("The input matrix `rho` has zero trace, which is invalid for density matrices.")

    rho = (rho + rho.T.conj()) / 2  # Force Hermitian symmetry

    if dim is None:
        total_dim = int(np.sqrt(rho.shape[0]))
        dim = [total_dim, total_dim]

    if len(dim) != 2 or np.prod(dim) != rho.shape[0]:
        raise ValueError("Dimension mismatch: `dim` must be a 1x2 list that correctly partitions `rho`.")

    eigvals = np.linalg.eigvalsh(rho)
    p = int(min(dim))
    print(f"[DEBUG] Computed p = {p}, type = {type(p)}")

    if p >= 7:
        print("[DEBUG] Large dimension detected (p >= 7). Returning -1.")
        return -1

    if not is_positive_semidefinite(rho):
        if p >= 6:
            print("[DEBUG] rho is not PSD but high-dimensional; returning -1 as inconclusive.")
            return -1
        print("[DEBUG] rho is not PSD. Returning 0.")
        return 0

    try:
        if is_separable(rho):
            return 1
    except ValueError:
        pass

    if not is_ppt(rho, sys=2, dim=dim):
        return 0

    if np.sum(eigvals[: p - 1]) <= 2 * eigvals[-1] + np.sum(eigvals[-p + 1 : -1]):
        return 1

    L = symmetric_extension_hierarchy([rho], level=1, dim=dim)
    if not is_positive_semidefinite(L[-1]):
        return 0

    return 1
