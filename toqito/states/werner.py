"""Werner state."""
from typing import List, Union

import itertools
import numpy as np

from toqito.perms import permutation_operator
from toqito.perms import swap_operator


def werner(dim: int, alpha: Union[float, List[float]]) -> np.ndarray:
    r"""
    Produce a Werner state [Wer89]_.

    A Werner state is a state of the following form

    .. math::

        \begin{equation}
            \rho_{\alpha} = \frac{1}{d^2 - d\alpha} \left(\mathbb{I} \otimes
            \mathbb{I} - \alpha S \right) \in \mathbb{C}^d \otimes \mathbb{C}^d.
        \end{equation}

    Yields a Werner state with parameter :code:`alpha` acting on :code:`(dim * dim)`- dimensional
    space. More specifically, :math:`\rho` is the density operator defined by
    :math:`(\mathbb{I} - `alpha` S)` (normalized to have trace 1), where :math:`\mathbb{I}` is the
    density operator and :math:`S` is the operator that swaps two copies of :code:`dim`-dimensional
    space (see swap and swap_operator for example).

    If :code:`alpha` is a vector with :math:`p!-1` entries, for some integer :math:`p > 1`, then a
    multipartite Werner state is returned. This multipartite Werner state is the normalization of
    I - `alpha(1)*P(2)` - ... - `alpha(p!-1)*P(p!)`, where P(i) is the operator that permutes p
    subsystems according to the i-th permutation when they are written in lexicographical order
    (for example, the lexicographical ordering when p = 3 is:
    `[1, 2, 3], [1, 3, 2], [2, 1,3], [2, 3, 1], [3, 1, 2], [3, 2, 1],`

    so P(4) in this case equals permutation_operator(dim, [2, 3, 1]).

    Examples
    ==========

    Computing the qutrit Werner state with :math:`\alpha = 1/2` can be done in :code:`toqito` as

    >>> from toqito.states import werner
    >>> werner(3, 1 / 2)
    [[ 0.06666667,  0.        ,  0.        ,  0.        ,  0.        ,
       0.        ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.13333333,  0.        , -0.06666667,  0.        ,
       0.        ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.13333333,  0.        ,  0.        ,
       0.        , -0.06666667,  0.        ,  0.        ],
     [ 0.        , -0.06666667,  0.        ,  0.13333333,  0.        ,
       0.        ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.06666667,
       0.        ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       0.13333333,  0.        , -0.06666667,  0.        ],
     [ 0.        ,  0.        , -0.06666667,  0.        ,  0.        ,
       0.        ,  0.13333333,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       -0.06666667,  0.        ,  0.13333333,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       0.        ,  0.        ,  0.        ,  0.06666667]]

    We may also compute multipartite Werner states in :code:`toqito` as well.

    >>> from toqito.states import werner
    >>> werner(2, [0.01, 0.02, 0.03, 0.04, 0.05])
    [[ 0.12179487,  0.        ,  0.        ,  0.        ,  0.        ,
       0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.12820513,  0.        ,  0.        , -0.00641026,
       0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.12179487,  0.        ,  0.        ,
       0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.12820513,  0.        ,
       0.        , -0.00641026,  0.        ],
     [ 0.        , -0.00641026,  0.        ,  0.        ,  0.12820513,
       0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       0.12179487,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        , -0.00641026,  0.        ,
       0.        ,  0.12820513,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       0.        ,  0.        ,  0.12179487]]

    References
    ==========
    .. [Wer89] R. F. Werner.
        Quantum states with Einstein-Podolsky-Rosen correlations admitting a
        hidden-variable model. Phys. Rev. A, 40(8):4277–4281. 1989

    :param dim: The dimension of the Werner state.
    :param alpha: Parameter to specify Werner state.
    :return: A Werner state of dimension :code:`dim`.
    """
    # The total number of permutation operators.
    if isinstance(alpha, float):
        n_fac = 2
    else:
        n_fac = len(alpha) + 1

    # Multipartite Werner state.
    if n_fac > 2:
        # Compute the number of parties from `len(alpha)`.
        n_var = n_fac
        # We won't actually go all the way to `n_fac`.
        for i in range(2, n_fac):
            n_var = n_var // i
            if n_var == i + 1:
                break
            if n_var < i:
                raise ValueError(
                    "InvalidAlpha: The `alpha` vector must contain"
                    " p!-1 entries for some integer p > 1."
                )

        # Done error checking and computing the number of parties -- now
        # compute the Werner state.
        perms = list(itertools.permutations(np.arange(n_var)))
        sorted_perms = np.argsort(perms, axis=1) + 1

        for i in range(2, n_fac):
            rho = np.identity(dim ** n_var) - alpha[i - 1] * permutation_operator(
                dim, sorted_perms[i, :], False, True
            )
        rho = rho / np.trace(rho)
        return rho
    # Bipartite Werner state.
    return (np.identity(dim ** 2) - alpha * swap_operator(dim, True)) / (dim * (dim - alpha))
