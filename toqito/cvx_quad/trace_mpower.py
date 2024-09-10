import numpy as np
from cvxpy import cvx


def trace_mpower(A: np.ndarray, t: float, C: np.ndarray = None) -> cvx.Expression:
    r"""
    Returns trace(C*A^t) where A and C are positive semidefinite.

    If C is None, then it is treated as an identity matrix.

    Examples
    ==========
    Later

    References
    ==========
    .. bibliography::
       :filter: docname in docnames

    A: np.ndarray
        Input positive definite matrix.

    t: float
      The exponent value. Must be between -1 and 2 inclusive.

    C: Optional[np.ndarray]
        Optional positive semidefinite matrix of the same size as A.
        If provided, the function computes trace(C*log(A)).
        Default is None, which corresponds to C being the identity matrix.
    """
    raise NotImplementedError
