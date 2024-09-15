import numpy as np
from cvxpy import cvx


def matrix_geo_mean_epi_cone(
    sz: list, t: float, iscplx: bool = False, fullhyp: bool = True
) -> tuple:
    r"""
    Returns a CVX tuple {A, B, T} of matrices constrained to satisfy A #_{t} B <= T,
    where A #_{t} B is the t-weighted geometric mean of A and B.

    The inequality "<=" is in the positive semidefinite order, and A, B, T are
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
        If a list is provided, it should be in the form [sz(1), sz(2), sz(3), ...].
        This will return an array of prod(sz[2:]) triples, where each triple has
        size sz[0]=sz[1] and is constrained to live in the operator relative
        entropy cone. Note that sz[0] and sz[1] must be equal in this case.
    t: float
        The parameter t should be in [-1, 0] or [1, 2].
    iscplx: bool, optional
        If True, the matrices are Hermitian. If False (default), the matrices are symmetric.
    fullhyp: bool, optional
        This parameter is not used and is here just for consistency with the hypo_cone function.
        matrix_geo_mean_epi_cone will always return a full epigraph cone.

    """
    raise NotImplementedError
