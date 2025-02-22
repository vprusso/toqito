import numpy as np
from cvxpy import cvx


def trace_logm(
    X: np.ndarray, C: np.ndarray = None, m: int = 3, k: int = 3, apx: int = 0
) -> cvx.Expression:
    r"""Trace of logarithm of a positive definite matrix.

    This function computes trace(logm(X)) where X is a positive definite matrix.
    It can also compute trace(C*logm(X)) where C is a positive semidefinite matrix
    of the same size as X.

    Examples
    ==========
    Later


    References
    ==========
    .. bibliography::
       :filter: docname in docnames

    X: np.ndarray
        Input positive definite matrix.
    C: Optional[np.ndarray]
        Optional positive semidefinite matrix of the same size as X.
        If provided, the function computes trace(C*logm(X)).
        Default is None, which corresponds to C being the identity matrix.
    m: int
        Number of quadrature nodes to use in the approximation.
        Default is 3.
    k: int
        Number of square-roots to take in the approximation.
        Default is 3.
    apx: int
        Indicates which approximation r of logm(X) to use:
        - apx = +1: Upper approximation (logm(X) <= r(X))
        - apx = -1: Lower approximation (r(X) <= logm(X))
        - apx =  0 (Default): Pade approximation (neither upper nor lower),
                              but slightly better accuracy than apx=+1 or -1.
    """
    raise NotImplementedError
