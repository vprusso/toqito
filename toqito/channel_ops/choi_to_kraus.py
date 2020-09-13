"""Compute a list of Kraus operators from the Choi matrix."""
from typing import List
import numpy as np

from toqito.matrix_ops import unvec


def choi_to_kraus(choi_mat: np.ndarray, tol: float = 1e-9) -> List[List[np.ndarray]]:
    r"""
    Compute a list of Kraus operators from the Choi matrix [Rigetti20]_.

    Note that unlike the Choi or natural representation of operators, the Kraus representation is
    *not* unique.

    This function has been adapted from [Rigetti20]_.

    Examples
    ========

    Convert the Choi operator

    See Also
    ========
    kraus_to_choi

    References
    ==========
    .. [Rigetti20] Forest Benchmarking (Rigetti).
        https://github.com/rigetti/forest-benchmarking

    :param choi_mat: a dim**2 by dim**2 choi matrix
    :param tol: optional threshold parameter for eigenvalues/kraus ops to be discarded
    :return: List of Kraus operators
    """
    eigvals, v_mat = np.linalg.eigh(choi_mat)
    return [
        np.lib.scimath.sqrt(eigval) * unvec(np.array([evec]).T)
        for eigval, evec in zip(eigvals, v_mat.T)
        if abs(eigval) > tol
    ]
