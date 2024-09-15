import numpy as np
from cvxpy import cvx


def matrix_geo_mean_hypo_cone(
    sz: list, t: float, iscplx: bool = False, fullhyp: bool = True
) -> tuple:
    r"""
    Returns a CVX tuple {A, B, T} of matrices constrained to satisfy A #_{t} B >= T,
    where A #_{t} B is the t-weighted geometric mean of A and B.

    The inequality ">=" is in the positive semidefinite order, and A, B, T are
    symmetric (Hermitian if iscplx=True) matrices of size sz.

    Examples
    ========
    Later

    References
    ==========
    .. bibliography::
         :filter: docname in docnames

    sz: list
        The size of the matrices. It can be a single integer or a list of integers.

    t: float
        The parameter t should be in [0, 1].

    iscplx: bool, optional
        If True, the matrices are Hermitian. If False (default), the matrices are symmetric.

    fullhyp: bool, optional
        If True (default), the full hypograph set is returned:
            hyp_t = {(A, B, T) : A #_{t} B >= T}
        If False, a convex set C_t is returned that satisfies:
            (A, B, A #_{t} B) \in C_t
            (A, B, T) \in C_t  =>  A #_{t} B >= T
        Setting fullhyp=False results in a slightly smaller SDP description.

    """
    raise NotImplementedError
