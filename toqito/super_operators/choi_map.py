import numpy as np
from toqito.states.max_entangled import max_entangled


def choi_map(a: int = 1, b: int = 1, c: int = 0) -> np.ndarray:
    """
    Produces the Choi map or one of its generalizations.

    The Choi map is a positive map on 3-by-3 matrices that is capable
    of detecting some entanglement that the transpose map is not.

    The standard Choi map defined with a=1, b=1, and c=0 is the
    Choi matrix of the positive map defined in [1]. Many of these
    maps are capable of detecting PPT entanglement.

    :param a: Default integer for standard Choi map.
    :param b: Default integer for standard Choi map.
    :param c: Default integer for standard Choi map.

    [1] S. J. Cho, S.-H. Kye, and S. G. Lee,
        Linear Alebr. Appl. 171, 213
        (1992).
    """
    psi = max_entangled(3, 0, 0)
    return np.diag([a+1, c, b, b, a+1, c, c, b, a+1]) - psi*psi.conj().T

