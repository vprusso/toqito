"""Produces an isotropic state."""
import numpy as np
from scipy.sparse import identity
from toqito.state.states.max_entangled import max_entangled


def isotropic(dim: int, alpha: float) -> np.ndarray:
    r"""
    Produce a isotropic state.

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

    References:
        [1] Horodecki, Michał, and Paweł Horodecki.
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
