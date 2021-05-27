"""Generalized Singlet state."""
import numpy as np

from toqito.perms import swap_operator
from toqito.matrices import iden

def singlet(dim: int) -> np.ndarray:
    r"""
    Produce a generalized Singlet state acting on two n-dimensional systems [Gsinglet]_.

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

    It is possible for us to consider higher dimensional Singlet states. For instance, we
    can consider the :math:`3`-dimensional Singlet state as follows:

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
    .. [Gsinglet] Adan Cabello.
      "N-particleN-level Singlet States:  Some Properties and Applications."
       Phys. Rev. Lett., 89, (2002): 100402.

    :param dim: The dimension of the generalized Singlet state.
    """
    return (iden(dim**2) - swap_operator([dim, dim]))/((dim**2)-dim)
    