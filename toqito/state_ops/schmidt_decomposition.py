"""Schmidt decomposition operation."""


import numpy as np


def schmidt_decomposition(
    rho: np.ndarray, dim: int | list[int] | np.ndarray = None, k_param: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute the Schmidt decomposition of a bipartite vector :cite:`WikiScmidtDecomp`.

    Examples
    ==========
    Consider the :math:`3`-dimensional maximally entangled state:

    .. math::
        u = \frac{1}{\sqrt{3}} \left( |000 \rangle + |111 \rangle + |222 \rangle \right).

    We can generate this state using the :code:`toqito` module as follows.

    >>> from toqito.states import max_entangled
    >>> max_entangled(3)
    [[0.57735027],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.57735027],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.57735027]]

    Computing the Schmidt decomposition of :math:`u`, we can obtain the corresponding singular
    values of :math:`u` as

    .. math::
        \frac{1}{\sqrt{3}} \left[1, 1, 1 \right]^{\text{T}}.

    >>> from toqito.states import max_entangled
    >>> from toqito.state_ops import schmidt_decomposition
    >>> singular_vals, u_mat, vt_mat = schmidt_decomposition(max_entangled(3))
    >>> singular_vals
    [[0.57735027]
     [0.57735027]
     [0.57735027]]
    >>> u_mat
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    >>> vt_mat
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If matrices are not of equal dimension.
    :param rho: A bipartite quantum state to compute the Schmidt decomposition of.
    :param dim: An array consisting of the dimensions of the subsystems (default gives subsystems
                equal dimensions).
    :param k_param: How many terms of the Schmidt decomposition should be computed (default is 0).
    :return: The Schmidt decomposition of the :code:`rho` input.

    """
    # If the input is provided as a matrix, compute the operator Schmidt decomposition.
    if len(rho.shape) == 2:
        if rho.shape[0] != 1 and rho.shape[1] != 1:
            return _operator_schmidt_decomposition(rho, dim, k_param)

    if dim is None:
        dim = np.round(np.sqrt(len(rho)))
    if isinstance(dim, list):
        dim = np.array(dim)

    # Allow the user to enter a single number for `dim`.
    if isinstance(dim, float):
        dim = np.array([dim, len(rho) / dim])
        dim[1] = np.round(dim[1])

    # Otherwise, use lots of Schmidt coefficients.
    u_mat, singular_vals, vt_mat = np.linalg.svd(rho.reshape(dim[::-1].astype(int), order="F"))

    # Otherwise, use lots of Schmidt coefficients.
    # After taking the transpose, the columns of `vt_mat` are actually the
    # (conjugate) singular vectors.  We do not take the conjugate because the
    # tensor product implementation does not take the conjugate either.  This is
    # not consistent with Wikipedia, which also takes the conjugate for complex
    # vectors.  Taking the conjugate would return right singular values that
    # need to be conjugated to reconstruct `rho`, which would be obviously
    # strange behavior.
    vt_mat = vt_mat.T

    if k_param > 0:
        u_mat = u_mat[:, :k_param]
        singular_vals = singular_vals[:k_param]
        vt_mat = vt_mat[:, :k_param]

    singular_vals = singular_vals.reshape(-1, 1)
    if k_param == 0:
        # Schmidt rank.
        r_param = np.sum(singular_vals > np.max(dim) * np.spacing(singular_vals[0]))
        # Schmidt coefficients.
        singular_vals = singular_vals[:r_param]
        u_mat = u_mat[:, :r_param]
        vt_mat = vt_mat[:, :r_param]

    return singular_vals, vt_mat, u_mat


def _operator_schmidt_decomposition(
    rho: np.ndarray, dim: int | list[int] | np.ndarray = None, k_param: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Calculate the Schmidt decomposition of an operator (matrix).

    Given an input `rho` provided as a matrix, determine its corresponding
    Schmidt decomposition.

    :raises ValueError: If matrices are not of equal dimension..
    :param rho: The matrix.
    :param dim: The dimension of the matrix
    :param k_param: The number of Schmidt coefficients to compute.
    :return: The Schmidt decomposition of the :code:`rho` input.
    """
    if dim is None:
        dim_x = rho.shape
        sqrt_dim = np.round(np.sqrt(dim_x))
        dim = np.array([[sqrt_dim[0], sqrt_dim[0]], [sqrt_dim[1], sqrt_dim[1]]])

    if isinstance(dim, list):
        dim = np.array(dim)

    # Allow the user to enter a single number for `dim` if `rho` is square.
    if isinstance(dim, int):
        dim = np.array([dim, len(rho) / dim])
        dim[1] = np.round(dim[1])

    if min(dim.shape) == 1 or len(dim.shape) == 1:
        dim = np.array([dim, dim])

    # Vectorize `rho` in a block ordering and compute singular values and vectors.
    rho = np.moveaxis(
        rho.reshape((int(dim[0, 0]), int(dim[0, 1]), int(dim[1, 0]), int(dim[1, 1]))),
        (1, 2),
        (2, 1),
    )
    singular_vals, u_mat, vt_mat = schmidt_decomposition(
        rho.reshape((rho.size, 1)), np.prod(dim, axis=0).astype(int), k_param
    )

    # Reshape columns of `u_mat` and `vt_mat` to form a list of left and right
    # singular operators of `rho`.  The singular operators are given along the
    # last axis for consistency with the Schmidt decomposition of a state
    # vector.
    u_mat = u_mat.reshape((int(dim[0, 0]), int(dim[1, 0]), len(singular_vals)))
    vt_mat = vt_mat.reshape((int(dim[0, 1]), int(dim[1, 1]), len(singular_vals)))

    return singular_vals, u_mat, vt_mat
