import numpy as np
from cvxpy import cvx


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
    raise NotImplementedError
