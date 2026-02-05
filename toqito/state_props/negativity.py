"""Calculates the negativity property of a quantum state."""

import numpy as np
from picos import partial_transpose

from toqito.matrix_ops import to_density_matrix


def negativity(rho: np.ndarray, dim: list[int] | int | None = None) -> float | np.floating:
    r"""Compute the negativity of a bipartite quantum state :footcite:`WikiNeg`.

    The negativity of a subsystem can be defined in terms of a density matrix :math:`\rho`:

    .. math::
        \mathcal{N}(\rho) \equiv \frac{||\rho^{\Gamma_A}||_1-1}{2}.

    Calculate the negativity of the quantum state :math:`\rho`, assuming that the two subsystems on
    which :math:`\rho` acts are of equal dimension (if the local dimensions are unequal, specify
    them in the optional :code:`dim` argument). The negativity of :math:`\rho` is the sum of the
    absolute value of the negative eigenvalues of the partial transpose of :math:`\rho`.

    Examples
    ==========

    Example of the negativity of density matrix of Bell state.

    .. jupyter-execute::

        from toqito.states import bell
        from toqito.state_props import negativity
        rho = bell(0) @ bell(0).conj().T
        negativity(rho)

    See Also
    ==========
    :py:func:`~toqito.state_props.log_negativity.log_negativity`

    References
    ==========
    .. footbibliography::



    :raises ValueError: If dimension of matrix is invalid.
    :param rho: A density matrix of a pure state vector.
    :param dim: The default has both subsystems of equal dimension.
    :return: A value between 0 and 1 that corresponds to the negativity of :math:`\rho`.

    """
    # Allow the user to input either a pure state vector or a density matrix.
    rho = to_density_matrix(rho)
    rho_dims = rho.shape
    round_dim = np.round(np.sqrt(rho_dims))

    if dim is None:
        dim_arr: np.ndarray = np.array([round_dim])
        dim_arr = dim_arr.T
    elif isinstance(dim, list):
        dim_arr = np.array(dim)
    elif isinstance(dim, int):
        dim_arr = np.array([dim, rho_dims[0] / dim])
        if abs(dim_arr[1] - np.round(dim_arr[1])) >= 2 * rho_dims[0] * np.finfo(float).eps:
            raise ValueError(
                "InvalidDim: If `dim` is a scalar, `rho` must be "
                "square and `dim` must evenly divide `len(rho)`. "
                "Please provide the `dim` array containing the "
                "dimensions of the subsystems."
            )
        dim_arr[1] = np.round(dim_arr[1])
    else:
        dim_arr = dim

    if np.prod(dim_arr) != rho_dims[0]:
        raise ValueError(
            "InvalidDim: Please provide local dimensions in the argument `dim` that match the size of `rho`."
        )

    dim_list = [int(x.item()) for x in dim_arr]

    # Compute the negativity.
    return (np.linalg.norm(partial_transpose(rho, [1], dim_list), ord="nuc") - 1) / 2
