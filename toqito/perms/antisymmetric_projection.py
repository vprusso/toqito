"""Antisymmetric projection operator produces an orthogonal projection onto an anti-symmetric subspace."""

from itertools import permutations

import numpy as np

from toqito.perms import perm_sign, permutation_operator


def antisymmetric_projection(dim: int, p_param: int = 2, partial: bool = False) -> np.ndarray:
    r"""Produce the projection onto the antisymmetric subspace :cite:`WikiAsymmOp`.

    Produces the orthogonal projection onto the anti-symmetric subspace of :code:`p_param` copies of
    :code:`dim`-dimensional space. If :code:`partial = True`, then the antisymmetric projection (PA) isn't the
    orthogonal projection itself, but rather a matrix whose columns form an orthonormal basis for the symmetric subspace
    (and hence the PA * PA' is the orthogonal projection onto the symmetric subspace.)

    Examples
    ==========

    The :math:`2`-dimensional antisymmetric projection with :math:`p=1` is given as
    :math:`2`-by-:math:`2` identity matrix

    .. math::
        A_{2,1} =
        \begin{pmatrix}
            1 & 0 \\
            0 & 1
        \end{pmatrix}.

    Using :code:`toqito`, we can see this gives the proper result.

    >>> from toqito.perms import antisymmetric_projection
    >>> antisymmetric_projection(2, 1)
    array([[1., 0.],
           [0., 1.]])

    When the :math:`p` value is greater than the dimension of the antisymmetric projection, this just gives the matrix
    consisting of all zero entries. For instance, when :math:`d = 2` and :math:`p = 3` we have that

    .. math::
        A_{2, 3} =
        \begin{pmatrix}
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
        \end{pmatrix}.

    Using :code:`toqito` we can see this gives the proper result.

    >>> from toqito.perms import antisymmetric_projection
    >>> antisymmetric_projection(2, 3)
    array([[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param dim: The dimension of the local systems.
    :param p_param: Default value of 2.
    :param partial: Default value of 0.
    :return: Projection onto the antisymmetric subspace.

    """
    dimp = dim**p_param

    if p_param == 1:
        return np.eye(dim)
    # The antisymmetric subspace is empty if `dim < p`.
    if dim < p_param:
        return np.zeros((dimp, dimp * (1 - partial)))

    p_list = np.array(list(permutations(np.arange(p_param ))))
    p_fac = p_list.shape[0]

    anti_proj = np.zeros((dimp, dimp))
    for j in range(p_fac):
        anti_proj += perm_sign(p_list[j, :]) * permutation_operator(dim * np.ones(p_param), p_list[j, :], False, True)
    anti_proj = anti_proj / p_fac

    if partial:
        anti_proj = np.array(np.linalg.qr(anti_proj))
    return anti_proj
