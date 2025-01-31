import numpy as np
from cvxpy import cvx
from scipy.linalg import fractional_matrix_power as fmp
import trace_mpower


def lieb_ando(A: np.ndarray, B: np.ndarray, K: np.ndarray, t: float) -> cvx.Expression:
    r"""
    Returns the value of Lieb's function.

    LIEB_ANDO(A,B,K,t) returns  trace(K' * A^{1-t} * K * B^t) where A and B
    are positive semidefinite matrices and K is an arbitrary matrix
    (possibly rectangular).

    Examples
    ==========
    Later

    References
    ==========
    .. bibliography::
       :filter: docname in docnames

    A: np.ndarray
      First positive semidefinite input matrix

    B: np.ndarray
      Second positive semidefinite input matrix

    K: np.ndarray
      An arbitrary matrix. K must have the same
      number of rows as A and the same number of columns as B.

    t: float
      The exponent value. Must be between -1 and 2 inclusive.
    """
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert B.shape[0] == B.shape[1], "B must be a square matrix"

    assert (
        K.shape[0] == A.shape[0] and K.shape[1] == B.shape[0]
    ), "K must have the same number of rows as A and the same number of columns as B"

    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return np.real(np.trace(K.conjugate().T @ fmp(A, 1 - t) @ K @ fmp(B, t)))
    elif isinstance(A, np.ndarray):
        KAK = K.conjugate().T @ fmp(A, 1 - t) @ K
        KAK = (KAK + KAK.conjugate().T) / 2
        return trace_mpower(B, 1 - t, KAK)
    elif isinstance(B, np.ndarray):
        KBK = K.conjugate().T @ fmp(B, 1 - t) @ K
        KBK = (KBK + KBK.conjugate().T) / 2
        return trace_mpower(A, 1 - t, KBK)
    elif A.is_affine() and B.is_affine():
        pass
    else:
        raise Exception("The input has to be an affine exression")
