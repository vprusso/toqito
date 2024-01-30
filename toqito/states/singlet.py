"""Generalized singlet state."""
import numpy as np

from toqito.perms import swap_operator


def singlet(dim: int) -> np.ndarray:
    r"""Produce a generalized singlet state acting on two n-dimensional systems :cite:`Cabello_2002_NParticle`.

    Examples
    ==========

    For :math:`n = 2` this generates the following matrix

    .. math::
        S = \frac{1}{2} \begin{pmatrix}
                        0 & 0 & 0 & 0 \\
                        0 & 1 & -1 & 0 \\
                        0 & -1 & 1 & 0 \\
                        0 & 0 & 0 & 0
                    \end{pmatrix}

    which is equivalent to :math:`|\phi_s \rangle \langle \phi_s |` where

    .. math::
        |\phi_s\rangle = \frac{1}{\sqrt{2}} \left( |01 \rangle - |10 \rangle \right)

    is the singlet state. This can be computed via :code:`toqito` as follows:

    >>> from toqito.states import singlet
    >>> dim = 2
    >>> singlet(dim)
        [[ 0. ,  0. ,  0. ,  0. ],
         [ 0. ,  0.5, -0.5,  0. ],
         [ 0. , -0.5,  0.5,  0. ],
         [ 0. ,  0. ,  0. ,  0. ]]

    It is possible for us to consider higher dimensional singlet states. For instance, we can consider the
    :math:`3`-dimensional Singlet state as follows:

    >>> from toqito.states import singlet
    >>> dim = 3
    >>> singlet(dim)
        [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ,  0.        ,  0.        ],
        [  0.        ,  0.16666667,  0.        , -0.16666667,  0.        ,
           0.        ,  0.        ,  0.        ,  0.        ],
        [  0.        ,  0.        ,  0.16666667,  0.        ,  0.        ,
           0.        , -0.16666667,  0.        ,  0.        ],
        [  0.        , -0.16666667,  0.        ,  0.16666667,  0.        ,
           0.        ,  0.        ,  0.        ,  0.        ],
        [  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ,  0.        ,  0.        ],
        [  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
           0.16666667,  0.        , -0.16666667,  0.        ],
        [  0.        ,  0.        , -0.16666667,  0.        ,  0.        ,
           0.        ,  0.16666667,  0.        ,  0.        ],
        [  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
           -0.16666667,  0.        ,  0.16666667,  0.        ],
        [  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ,  0.        ,  0.        ]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param dim: The dimension of the generalized singlet state.
    :return: The singlet state of dimension `dim`.

    """
    return (np.identity(dim ** 2) - swap_operator([dim, dim])) / ((dim ** 2) - dim)
