"""Computes the entanglement of formation of a bipartite quantum state."""
from typing import List, Union

import numpy as np
import scipy

from toqito.channels import partial_trace
from toqito.state_props import concurrence, von_neumann_entropy


def entanglement_of_formation(rho: np.ndarray, dim: Union[List[int], int] = None) -> float:
    r"""
    Compute entanglement-of-formation of a bipartite quantum state [WikEOF]_.

    Entanglement-of-formation is the entropy of formation of the bipartite
    quantum state :code:`rho`. Note that this function currently only supports
    :code:`rho` being a pure state or a 2-qubit state: it is not known how to
    compute the entanglement-of-formation of higher-dimensional mixed states.

    This function was adapted from QETLAB.

    Examples
    ==========

    Compute the entanglement-of-formation of a Bell state.

    Let :math:`u = \frac{1}{\sqrt{2}} \left(|00\rangle + |11\rangle \right)`
    and let

    .. math::
        \rho = uu^* = \frac{1}{2}\begin{pmatrix}
                                    1 & 0 & 0 & 1 \\
                                    0 & 0 & 0 & 0 \\
                                    0 & 0 & 0 & 0 \\
                                    1 & 0 & 0 & 1
                                 \end{pmatrix}.

    The entanglement-of-formation of :math:`\rho` is equal to 1.

    >>> from toqito.state_props import entanglement_of_formation
    >>> from toqito.states import bell
    >>>
    >>> u_vec = bell(0)
    >>> rho = u_vec * u_vec.conj().T
    >>> entanglement_of_formation(rho)
    1

    References
    ==========
    .. [WikEOF] Quantiki: Entanglement-of-formation
        https://www.quantiki.org/wiki/entanglement-formation

    :param rho: A matrix or vector.
    :param dim: The default has both subsystems of equal dimension.
    :return: A value between 0 and 1 that corresponds to the
             entanglement-of-formation of :code:`\rho`.
    """
    dim_x, dim_y = rho.shape
    round_dim = int(np.round(np.sqrt(max(dim_x, dim_y))))
    eps = np.finfo(float).eps

    if dim is None:
        dim = round_dim

    # User can specify dimension as integer.
    if isinstance(dim, int):
        dim = np.array([dim, max(dim_x, dim_y) / dim], dtype=int)
        if abs(dim[1] - np.round(dim[1])) >= 2 * max(dim_x, dim_y) * eps:
            raise ValueError(
                "Invalid dimension: If `dim` is provided as a "
                "scalar, `dim` must evenly divide `len(rho)`."
            )
        dim[1] = np.round(dim[1])

    if np.prod(dim) != max(dim_x, dim_y):
        raise ValueError(
            "Invalid dimension: Please provide local dimensions " "that match the size of `rho`."
        )

    # If :code:`rho` is a rank-1 density matrix, turn it into a vector instead
    # so we can compute the entanglement-of-formation easily.
    tmp_rho = scipy.linalg.orth(rho)
    if dim_x == dim_y and tmp_rho.shape[1] == 1:
        rho = tmp_rho
        dim_y = 1

    # Start computing entanglement-of-formation.
    if min(dim_x, dim_y) == 1:
        rho = rho[:]
        return von_neumann_entropy(partial_trace(rho * rho.conj().T, 2, dim))

    # Case: :code:`rho` is a density matrix.
    if dim_x == dim_y:
        # In the two-qubit case, we know how to compute the
        # entanglement-of-formation exactly.
        if dim_x == 4:
            rho_c = concurrence(rho)

            rho_c1 = (1 + np.sqrt(1 - rho_c ** 2)) / 2
            rho_c2 = (1 - np.sqrt(1 - rho_c ** 2)) / 2

            rho_c1_log2 = 0 if rho_c1 == 0 else np.log2(rho_c1)
            rho_c2_log2 = 0 if rho_c2 == 0 else np.log2(rho_c2)

            return -rho_c1 * rho_c1_log2 - rho_c2 * rho_c2_log2
        raise ValueError(
            "Invalid dimension: It is presently only known how to compute "
            "the entanglement-of-formation for two-qubit states and pure "
            "states."
        )
    raise ValueError("Invalid dimension: `rho` must be either a vector or " "square matrix.")
