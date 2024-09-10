import numpy as np
from cvxpy import cvx


def quantum_rel_entr(
    A: np.ndarray, B: np.ndarray, m: int = 3, k: int = 3, apx: int = 0
) -> cvx.Expression:
    r"""
    Returns trace(A*(logm(A)-logm(B))).

    A and B are positive semidefinite matrices such that \im(A) \subseteq \im(B)
    (otherwise the function evaluates to infinity). Note this function uses
    logarithm base e (and not base 2!).

    Examples
      ==========
      Later

    References
      ==========
      .. bibliography::
         :filter: docname in docnames

    A: np.ndarray
      The first input positive semidefinite matrix.

    B: np.ndarray
      The second input positive semidefinite matrix.

    m: int
        Number of quadrature nodes to use in the approximation.
        Default is 3.
    k: int
        Number of square-roots to take in the approximation.
        Default is 3.
    apx: int
        Indicates which approximation r of the relative entropy function to use:
        - apx = +1: Upper approximation (D(A|B) <= r(A,B))
        - apx = -1: Lower approximation (r(A,B) <= D(A|B))
        - apx =  0 (Default): Pade approximation (neither upper nor lower),
                              but slightly better accuracy than apx=+1 or -1.
    """
    raise NotImplementedError
