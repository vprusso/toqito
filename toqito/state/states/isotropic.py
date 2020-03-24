"""Produces an isotropic state."""
import numpy as np
from scipy.sparse import identity
from toqito.state.states.max_entangled import max_entangled


def isotropic(dim: int, alpha: float) -> np.ndarray:
    """
    Produce a Isotropic state.

    Returns the isotropic state with parameter `alpha` acting on
    (`dim`-by-`dim`)-dimensional space. More specifically, the state is the
    density operator defined by `(1-alpha)*I(dim)/dim**2 + alpha*E`, where I is
    the identity operator and E is the projection onto the standard
    maximally-entangled pure state on two copies of `dim`-dimensional space.

    References:
    [1] N. Gisin. Hidden quantum nonlocality revealed by local filters.
        (http://dx.doi.org/10.1016/S0375-9601(96)80001-6). 1996.

    :param dim: The local dimension.
    :param alpha: The parameter of the isotropic state.
    :return: Isotropic state.
    """
    # Compute the isotropic state.
    psi = max_entangled(dim, True, False)
    return (1 - alpha) * identity(dim**2)/dim**2 + alpha*psi*psi.conj().T/dim
