"""Produces an isotropic state."""
import numpy as np
from scipy.sparse import identity
from toqito.states.states.max_entangled import max_entangled


def isotropic(dim: int, alpha: float) -> np.ndarray:
    r"""
    Produce a isotropic state [10]_.

    Returns the isotropic state with parameter `alpha` acting on
    (`dim`-by-`dim`)-dimensional space. More specifically, the state is the
    density operator defined by `(1-alpha)*I(dim)/dim**2 + alpha*E`, where I is
    the identity operator and E is the projection onto the standard
    maximally-entangled pure state on two copies of `dim`-dimensional space.

    The isotropic state has the following form

    .. math::

        \begin{equation}
            \rho_{\alpha} = \frac{1 - \alpha}{d^2} \mathbb{I} \otimes
            \mathbb{I} + \alpha |\psi_+ \rangle \langle \psi_+ | \in
            \mathbb{C}^d \otimes \mathbb{C}^2
        \end{equation}

    where :math:`|\psi_+ \rangle = \frac{1}{\sqrt{d}} \sum_j |j \rangle \otimes
    |j \rangle` is the maximally entangled state.

    Examples
    ==========

    To generate the isotropic state with parameter :math:`\alpha=1/2`, we can
    make the following call to `toqito` as

    >>> from toqito.states.states.isotropic import isotropic
    >>> isotropic(3, 1 / 2).toarray()
    array([[0.22222222, 0.        , 0.        , 0.        , 0.16666667,
            0.        , 0.        , 0.        , 0.16666667],
           [0.        , 0.05555556, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.05555556, 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.05555556, 0.        ,
            0.        , 0.        , 0.        , 0.        ],
           [0.16666667, 0.        , 0.        , 0.        , 0.22222222,
            0.        , 0.        , 0.        , 0.16666667],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.05555556, 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.05555556, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.05555556, 0.        ],
           [0.16666667, 0.        , 0.        , 0.        , 0.16666667,
            0.        , 0.        , 0.        , 0.22222222]])

    References
    ==========
    .. [10] Horodecki, Michał, and Paweł Horodecki.
        "Reduction criterion of separability and limits for a class of
        distillation protocols." Physical Review A 59.6 (1999): 4206.

    :param dim: The local dimension.
    :param alpha: The parameter of the isotropic state.
    :return: Isotropic state.
    """
    # Compute the isotropic state.
    psi = max_entangled(dim, True, False)
    return (1 - alpha) * identity(
        dim ** 2
    ) / dim ** 2 + alpha * psi * psi.conj().T / dim
