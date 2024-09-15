import numpy as np
from cvxpy import cvx


def quantum_cond_entr(rho: np.ndarray, dim: list, sys: int = 1) -> cvx.Expression:
    r"""
    Computes the quantum conditional entropy.

    If rho is a symmetric (or Hermitian) matrix of size na*nb, then:
    - quantum_cond_entr(rho, [na, nb]) returns H(A|B)
    - quantum_cond_entr(rho, [na, nb], 2) returns H(B|A)

    This function is a concave function of rho.


    Examples
    ========
    Later

    References
    ==========
    .. bibliography::
         :filter: docname in docnames

    rho: np.ndarray
        The input symmetric (or Hermitian) matrix.
    dim: list
        A list containing the dimensions [na, nb] of the subsystems.
    sys: int
        The subsystem for which the conditional entropy is computed.
        - sys = 1 (default): Computes H(A|B)
        - sys = 2: Computes H(B|A)
    """
    raise NotImplementedError
