"""Check if a quantum state is absolutely PPT."""

import numpy as np

from toqito.matrix_props.is_density import is_density
from toqito.state_opt.symmetric_extension_hierarchy import symmetric_extension_hierarchy
from toqito.state_props.in_separable_ball import in_separable_ball
from toqito.state_props.is_ppt import is_ppt


def is_abs_ppt(rho: np.ndarray, dim: list[int] | None = None, max_constraints: int = 2612) -> bool | None:
    r"""Determine whether a quantum state is absolutely PPT.

    Examples
    =========
    A maximally mixed state on 2 x 2:

    >>> import numpy as np
    >>> rho = np.array([[0.5, 0.0], [0.0, 0.5]])
    >>> is_abs_ppt(rho, [2, 1])
    True

    Bell state (entangled, not absolutely PPT):

    >>> from toqito.states import bell
    >>> bell_state = bell(0) @ bell(0).conj().T
    >>> is_abs_ppt(bell_state, [2, 2])
    False

    High-dimensional identity matrix:

    >>> rho_7 = np.eye(49) / 49
    >>> is_abs_ppt(rho_7, [7, 7])
    True

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param rho: Density matrix to be tested.
    :param dim: A list of two integers representing bipartite dimensions. If `None`, assume equal split.
    :param max_constraints: Maximum number of SDP constraints for symmetric extension checks.
    :raises ValueError: If `rho` is invalid as a density matrix.
    :return: `True` if absolutely PPT, `False` if not, or `None` if inconclusive.

    """
    if not is_density(rho):
        raise ValueError("Input `rho` is not a valid density matrix.")

    rho = (rho + rho.conj().T) / 2
    d = rho.shape[0]
    if dim is None:
        s = int(np.sqrt(d))
        if s * s != d:
            raise ValueError("Unable to infer valid bipartite dimensions; please specify `dim` explicitly.")
        dim = [s, s]
    if len(dim) != 2 or dim[0] * dim[1] != d:
        raise ValueError("`dim` must match the shape of `rho`.")

    lam = np.linalg.eigvalsh(rho)
    p = min(dim)

    if in_separable_ball(np.sort(lam)[::-1]):
        return True
    if not is_ppt(rho, sys=2, dim=dim):
        return False

    lam_sorted = np.sort(lam)[::-1]
    if np.sum(lam_sorted[: p - 1]) <= 2 * lam_sorted[-1] + np.sum(lam_sorted[-(p - 1) : -1]):
        return True

    try:
        L = symmetric_extension_hierarchy([rho], level=1, dim=dim, max_constraints=max_constraints)
        if np.any(np.linalg.eigvals(L[-1]) < 0):
            return False
        return True if p <= 6 else None
    except Exception:
        return None
