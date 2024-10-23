"""Calculates the logarithmic negativity property of a quantum state."""

import numpy as np
from picos import partial_transpose

from toqito.matrix_ops import to_density_matrix


def log_negativity(rho: np.ndarray, dim: list[int] | int = None) -> float:
    r"""Compute the log-negativity of a bipartite quantum state :cite:`WikiNeg`.

    The log-negativity of a subsystem can be defined in terms of a density matrix :math:`\rho`:

    .. math::
        E_\mathcal{N}(\rho) \equiv \text{log}_2\left( ||\rho^{\Gamma_A}||_1 \right).

    Calculate the log-negativity of the quantum state :math:`\rho`, assuming that the two subsystems
    on which :math:`\rho` acts are of equal dimension (if the local dimensions are unequal, specify
    them in the optional :code:`dim` argument).

    Examples
    ==========

    Example of the log-negativity of density matrix of Bell state.

    >>> from toqito.states import bell
    >>> from toqito.state_props import log_negativity
    >>> rho = bell(0) @ bell(0).conj().T
    >>> log_negativity(rho)
    np.float64(0.9999999999999997)

    See Also
    ==========
    negativity

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If the input matrix is not a density matrix.
    :param rho: A density matrix of a pure state vector.
    :param dim: The default has both subsystems of equal dimension.
    :return: A positive value that corresponds to the logarithmic negativity of :math:`\rho`.

    """
    # Allow the user to input either a pure state vector or a density matrix.
    rho = to_density_matrix(rho)
    rho_dims = rho.shape
    round_dim = np.round(np.sqrt(rho_dims))

    if dim is None:
        dim = np.array([round_dim])
        dim = dim.T
    if isinstance(dim, list):
        dim = np.array(dim)

    # Allow the user to enter a single number for dim.
    if isinstance(dim, int):
        dim = np.array([dim, rho_dims[0] / dim])
        if abs(dim[1] - np.round(dim[1])) >= 2 * rho_dims[0] * np.finfo(float).eps:
            raise ValueError(
                "InvalidDim: If `dim` is a scalar, `rho` must be "
                "square and `dim` must evenly divide `len(rho)`. "
                "Please provide the `dim` array containing the "
                "dimensions of the subsystems."
            )
        dim[1] = np.round(dim[1])

    if np.prod(dim) != rho_dims[0]:
        raise ValueError(
            "InvalidDim: Please provide local dimensions in the argument `dim` that match the size of `rho`."
        )

    dim = [int(x.item()) for x in dim]

    # Compute the log-negativity.
    return np.log2(np.linalg.norm(partial_transpose(rho, [1], dim), ord="nuc"))
