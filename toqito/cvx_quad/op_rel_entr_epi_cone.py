import numpy as np
import cvxpy as cvx


def op_rel_entr_epi_cone(
    sz: list,
    iscplx: bool = False,
    m: int = 3,
    k: int = 3,
    e: np.ndarray = None,
    apx: int = 0,
) -> tuple:
    """
    Returns a CVX triple {X, Y, TAU} of matrices constrained to satisfy:
        X^{1/2} * logm(X^{1/2} * Y^{-1} * X^{1/2}) * X^{1/2} <= TAU
    where:
        - The inequality "<=" is in the positive semidefinite order.
        - X, Y, TAU are symmetric (Hermitian if iscplx=True) matrices of size sz.
        - logm is the matrix logarithm (in base e).

    Examples
    ========
    Later

    References
    ==========
    .. bibliography::
         :filter: docname in docnames

    sz : list
        The size of the matrices. It can be a single integer or a list of integers.
        If an array [sz(1) sz(2) sz(3) ...] is provided, it will return an array of
        prod(sz(3:end)) triples, where each triple has size sz(1)=sz(2) and is
        constrained to live in the operator relative entropy cone. Note that sz(1)
        and sz(2) must be equal in this case.

    iscplx : bool, optional
        If True, the matrices are Hermitian. If False (default), the matrices are symmetric.

    m : int, optional
        The number of quadrature nodes to use for the approximation. Default is 3.

    k : int, optional
        The number of square-roots to take for the approximation. Default is 3.

    e : np.ndarray, optional
        If provided, it should be a matrix of size n x r (where n = sz(1)). The
        returned tuple(s) {X, Y, TAU} is then constrained to satisfy:
            e' * (D_{op}(X||Y)) * e <= TAU
        Note that TAU here is of size r x r. The default case corresponds to e = eye(n).
        When r is small compared to n, this can be helpful to reduce the size of small
        LMIs from 2n x 2n to (n+r) x (n+r).

    apx : int, optional
        Indicates which approximation of the logarithm to use. Possible values:
        - apx = +1: Upper approximation on D_{op} [inner approximation of the
                    operator relative entropy cone].
        - apx = -1: Lower approximation on D_{op} [outer approximation of the
                    operator relative entropy cone].
        - apx = 0 (Default): Pade approximation (neither upper nor lower) but
                             slightly better accuracy than apx = +1 or -1.

    """
    raise NotImplementedError
