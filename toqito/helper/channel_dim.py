"""Channel dimensions."""


import itertools

import numpy as np


def channel_dim(
    phi: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    allow_rect: bool = True,
    dim: int | list[int] | np.ndarray = None,
    compute_env_dim: bool = True,
) -> tuple[np.ndarray | int]:
    """Compute the input, output, and environment dimensions of a channel.

    This function returns the dimensions of the input, output, and environment spaces of
    input channel, in that order. Input and output dimensions are both 1-by-2 vectors
    containing the row and column dimensions of their spaces. The enviroment dimension
    is always a scalar, and it is equal to the number of Kraus operators of PHI (if PHI is
    provided as a Choi matrix then enviroment dimension is the *minimal* number of Kraus
    operators of any representation of PHI).

    Input DIM should provided if and only if PHI is a Choi matrix with unequal input and
    output dimensions (since it is impossible to determine the input and output dimensions
    from the Choi matrix alone). If ALLOW_RECT is false and PHI acts on non-square matrix
    spaces, an error will be produced. If PHI maps M_{r,c} to M_{x,y} then DIM should be the
    2-by-2 matrix [[r,x], [c,y]]. If PHI maps M_m to M_n, then DIM can simply be the vector
    [m,n]. If ALLOW_RECT is false then returned input and output dimensions will be scalars
    instead of vectors. If COMPUTE_ENV_DIM is false and the PHI is a Choi matrix we avoid
    computing the rank of the Choi matrix.

    This functions was adapted from QETLAB :cite:`QETLAB_link`.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param phi: A superoperator. It should be provided either as a Choi matrix,
                or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
    :param allow_rect: A flag indicating that the input and output spaces of PHI can be non-square (default True).
    :param dim: A scalar, vector or matrix containing the input and output dimensions of PHI.
    :param compute_env_dim: A flag indicating whether we compute the enviroment dimension.
    :return: The input, output, and environment dimensions of a channel.

    """
    dim_in = np.zeros(2, dtype=int)
    dim_out = np.zeros(2, dtype=int)

    if isinstance(phi, list):
        sz_phi_op = [len(phi), len(phi[0])]

        # Map is completely positive if input is given as:
        # 1. [K1, K2, .. Kr]
        # 2. [[K1], [K2], .. [Kr]]
        # 3. [[K1, K2, .. Kr]] and r > 2
        is_cpt = False
        if isinstance(phi[0], list) and (
            sz_phi_op[1] == 1 or (sz_phi_op[0] == 1 and sz_phi_op[1] > 2)
        ):
            # get a flat list of Kraus operators.
            phi = list(itertools.chain(*phi))
            is_cpt = True

        dim_e = len(phi)
        if isinstance(phi[0], np.ndarray):
            dim_out[0], dim_in[0] = phi[0].shape
            # input and output are squares.
            dim_in[1] = dim_in[0]
            dim_out[1] = dim_out[0]
            is_cpt = True
        else:
            dim_out[0], dim_in[0] = phi[0][0].shape
            dim_out[1], dim_in[1] = phi[0][1].shape

        if dim is None:
            dim = np.vstack([dim_in, dim_out]).T
        dim = _expand_dim(dim)

        # Now do some error checking.
        if (dim_in[0] != dim_in[1] or dim_out[0] != dim_out[1]) and not allow_rect:
            raise ValueError("The input and output spaces of PHI must be square.")

        if np.any(dim != np.vstack([dim_in, dim_out]).T):
            raise ValueError(
                "The dimensions of PHI do not match those provided in the DIM argument."
            )

        if (is_cpt and any(k_mat.shape != (dim[0, 1], dim[0, 0]) for k_mat in phi)) or (
            not is_cpt
            and any(
                k_mat[0].shape != (dim[0, 1], dim[0, 0]) or k_mat[1].shape != (dim[1, 1], dim[1, 0])
                for k_mat in phi
            )
        ):
            raise ValueError("The Kraus operators of PHI do not all have the same size.")

    # If Phi is a Choi matrix, the dimensions are a bit more of a pain: we have
    # to guess a bit if the input and output dimensions are different.
    else:
        # Try to guess input and output dims.
        rows, cols = phi.shape
        dim_in = np.array([int(np.round(np.sqrt(rows))), int(np.round(np.sqrt(cols)))])
        dim_out = dim_in

        if dim is None:
            dim = np.vstack([dim_in, dim_out]).T
        dim = _expand_dim(dim)

        if dim[0, 0] * dim[0, 1] != rows or dim[1, 0] * dim[1, 1] != cols:
            raise ValueError(
                "If the input and output dimensions are unequal and PHI is provided "
                "as a Choi matrix, the optional argument DIM must be specified "
                "(and its dimensions must agree with PHI)."
            )

        if (dim[0, 0] != dim[1, 0] or dim[0, 1] != dim[1, 1]) and not allow_rect:
            raise ValueError("The input and output spaces of PHI must be square.")

        # environment dimension is the rank of the Choi matrix
        dim_e = None
        if compute_env_dim:
            dim_e = np.linalg.matrix_rank(phi)

    # Finally, put `dim` back into `dim_in` and `dim_out`.
    if allow_rect:
        dim_in = np.array([dim[0, 0], dim[1, 0]])
        dim_out = np.array([dim[0, 1], dim[1, 1]])
    else:
        dim_in = dim[0, 0]
        dim_out = dim[0, 1]

    return (dim_in, dim_out, dim_e)


def _expand_dim(dim):
    # user just entered a single number for DIM
    if isinstance(dim, int):
        return np.array([[dim, dim], [dim, dim]])

    dim = np.array(dim)
    # user entered a full 2-by-2 matrix for DIM
    if dim.shape == (2, 2):
        return dim

    dim = dim.ravel()
    # user entered a 2-dimensional vector for DIM
    if dim.shape == (2,):
        return np.vstack([dim, dim])

    raise ValueError("The dimensions must be provided in a matrix no larger than 2-by-2.")
