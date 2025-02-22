import numpy as np
from cvxpy import cvx
from scipy.linalg import fractional_matrix_power as fmp


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
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    if C is None:
        C = np.eye(A.shape[0])
    else:
        C = (C + C.conjugate().T) / 2
        if np.any(np.linalg.eigvals(C) < -1 * 1e-6):
            raise Exception("C must be positive semidefinite")
    if isinstance(A, np.ndarray):
        return np.trace(np.dot(C, fmp(A, t)))
    elif A.is_affine():
        if t < -1 or t > 2:
            raise Exception("t must be between -1 and 2")
        pass
    else:
        raise Exception("The input has to be an affine expression")
