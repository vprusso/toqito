"""Is matrix a density matrix."""
import numpy as np
from toqito.matrix_props import is_positive_semidefinite


def is_density(mat: np.ndarray) -> bool:
    r"""
    Check if matrix is a density matrix [WikDensity]_.

    A matrix is a density matrix if its trace is equal to one and it has the
    property of being positive semidefinite (PSD).

    Examples
    ==========

    Consider the Bell state:

    .. math::
       u = \frac{1}{\sqrt{2}} |00 \rangle + \frac{1}{\sqrt{2}} |11 \rangle.

    Constructing the matrix :math:`\rho = u u^*` defined as

    .. math::
        \rho = \frac{1}{2} \begin{pmatrix}
                                1 & 0 & 0 & 1 \\
                                0 & 0 & 0 & 0 \\
                                0 & 0 & 0 & 0 \\
                                1 & 0 & 0 & 1
                           \end{pmatrix}

    our function indicates that this is indeed a density operator as the trace
    of :math:`\rho` is equal to :math:`1` and the matrix is positive
    semidefinite:

    >>> from toqito.matrix_props import is_density
    >>> from toqito.states import bell
    >>> import numpy as np
    >>> rho = bell(0) * bell(0).conj().T
    >>> is_density(rho)
    True

    Alternatively, the following example matrix :math:`\sigma` defined as

    .. math::
        \sigma = \frac{1}{2} \begin{pmatrix}
                                1 & 2 \\
                                3 & 1
                             \end{pmatrix}

    does satisfy :math:`\text{Tr}(\sigma) = 1`, however fails to be positive
    semidefinite, and is therefore not a density operator. This can be
    illustrated using :code:`toqito` as follows.

    >>> from toqito.matrix_props import is_density
    >>> from toqito.states import bell
    >>> import numpy as np
    >>> sigma = 1/2 * np.array([[1, 2], [3, 1]])
    >>> is_density(sigma)
    False

    References
    ==========
    .. [WikDensity] Wikipedia: Density matrix
        https://en.wikipedia.org/wiki/Density_matrix

    :param mat: Matrix to check.
    :return: Return `True` if matrix is a density matrix, and `False`
             otherwise.
    """
    return is_positive_semidefinite(mat) and np.isclose(np.trace(mat), 1)
