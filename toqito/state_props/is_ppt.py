"""Check if violates the PPT criterion."""


import numpy as np

from toqito.channels import partial_transpose
from toqito.matrix_props import is_positive_semidefinite


def is_ppt(mat: np.ndarray, sys: int = 2, dim: int | list[int] = None, tol: float = None) -> bool:
    r"""Determine whether or not a matrix has positive partial transpose :cite:`WikiPeresHorodecki`.

    Yields either :code:`True` or :code:`False`, indicating that :code:`mat` does or does not have
    positive partial transpose (within numerical error). The variable :code:`mat` is assumed to act
    on bipartite space.

    For shared systems of :math:`2 \otimes 2` or :math:`2 \otimes 3`, the PPT criterion serves as a
    method to determine whether a given state is entangled or separable. Therefore, for systems of
    this size, the return value :code:`True` would indicate that the state is separable and a value
    of :code:`False` would indicate the state is entangled.

    Examples
    ==========

    Consider the following matrix

    .. math::
        X =
        \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
        \end{pmatrix}.

    This matrix trivially satisfies the PPT criterion as can be seen using the
    :code:`toqito` package.

    >>> from toqito.state_props import is_ppt
    >>> import numpy as np
    >>> mat = np.identity(9)
    >>> is_ppt(mat)
    True

    Consider the following Bell state:

    .. math::
        u = \frac{1}{\sqrt{2}}\left( |01 \rangle + |10 \rangle \right).

    For the density matrix :math:`\rho = u u^*`, as this is an entangled state
    of dimension :math:`2`, it will violate the PPT criterion, which can be seen
    using the :code:`toqito` package.

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_ppt
    >>> rho = bell(2) * bell(2).conj().T
    >>> is_ppt(rho)
    False

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param mat: A square matrix.
    :param sys: Scalar or vector indicating which subsystems the transpose
                should be applied on.
    :param dim: The dimension is a vector containing the dimensions of the
                subsystems on which :code:`mat` acts.
    :param tol: Tolerance with which to check whether `mat` is PPT.
    :return: Returns :code:`True` if :code:`mat` is PPT and :code:`False` if
             not.

    """
    eps = np.finfo(float).eps

    sqrt_rho_dims = np.round(np.sqrt(list(mat.shape)))
    sqrt_rho_dims = np.int_(sqrt_rho_dims)

    if dim is None:
        dim = [
            [sqrt_rho_dims[0], sqrt_rho_dims[0]],  # pylint: disable=unsubscriptable-object
            [sqrt_rho_dims[1], sqrt_rho_dims[1]],  # pylint: disable=unsubscriptable-object
        ]
    if tol is None:
        tol = np.sqrt(eps)
    return is_positive_semidefinite(partial_transpose(mat, [sys - 1], dim), tol)
