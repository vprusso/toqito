import numpy as np
from cvxpy import cvx


def quantum_entr(X: np.ndarray, m: int = 3, k: int = 3, apx: int = 0) -> cvx.Expression:
    r"""
    Returns the quantum (Von Neumann) entropy of X.

    The quantum entropy is given by -trace(X*logm(X)) where the logarithm is
    base e (and not base 2!). X must be a positive semidefinite matrix. The
    implementation uses the operator relative entropy.

    Examples
    ==========
    Later

    References
    ==========
    .. bibliography::
       :filter: docname in docnames

    X: np.ndarray
        Input positive definite matrix.
    m: int
        Number of quadrature nodes to use in the approximation.
        Default is 3.
    k: int
        Number of square-roots to take in the approximation.
        Default is 3.
    apx: int
        Indicates which approximation r of the Von Neumann entropy to use:
        - apx = +1: Upper approximation of entropy (H(X) <= r(X))
        - apx = -1: Lower approximation (r(X) <= H(X))
        - apx =  0 (Default): Pade approximation (neither upper nor lower),
                              but slightly better accuracy than apx=+1 or -1.

    """
    raise NotImplementedError
