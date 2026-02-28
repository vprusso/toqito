"""Checks if the matrix is block positive."""

import numpy as np

from toqito.matrix_props.is_hermitian import is_hermitian
from toqito.matrix_props.is_positive_semidefinite import is_positive_semidefinite
from toqito.matrix_props.sk_norm import sk_operator_norm


def is_block_positive(
    mat: np.ndarray,
    k: int = 1,
    dim: int | list[int] | None = None,
    effort: int = 2,
    rtol: float = 1e-5,
) -> bool | RuntimeError:
    r"""Check if matrix is block positive [@Johnston_2012_Norms].

    Examples:
        The swap operator is always block positive, since it is the Choi
        matrix of the transpose map.

        ```python exec="1" source="above"
        from toqito.perms.swap_operator import swap_operator
        from toqito.matrix_props.is_block_positive import is_block_positive

        mat = swap_operator(3)

        print(is_block_positive(mat=mat))
        ```


        However, it's not 2 - block positive.

        ```python exec="1" source="above"
        from toqito.perms.swap_operator import swap_operator
        from toqito.matrix_props.is_block_positive import is_block_positive

        mat = swap_operator(3)

        print(is_block_positive(mat=mat, k=2))
        ```

    Raises:
        RuntimeError: Unable to determine k-block positivity. Please consider increasing the relative tolerance or the
            effort level.

    Args:
        mat: A bipartite Hermitian operator.
        k: A positive integer indicating that the function should determine whether or not the input operator is k-block
            positive, i.e., whether or not it remains nonnegative under left and right multiplication by vectors with
            Schmidt rank <= k (default 1).
        dim: The dimension of the two sub-systems. By default it's assumed to be equal.
        effort: An integer value indicating the amount of computation you want to devote to determine block positivity
            before giving up.
        rtol: The relative tolerance parameter (default 1e-05).

    Returns:
        Return `True` if matrix is k-block positive definite, `False` if not, or raise a runtime error if we are unable
        to determine whether or not the operator is block positive.

    """
    if not is_hermitian(mat):
        return False

    dim_xy = mat.shape[0]
    # Set default dimension if none was provided.
    if dim is None:
        dim_val = int(np.round(np.sqrt(dim_xy)))
    elif isinstance(dim, int):
        dim_val = dim
    else:
        dim_val = None

    # Allow the user to enter in a single integer for dimension.
    if dim_val is not None:
        dim_arr = np.array([dim_val, dim_xy / dim_val])
        dim_arr[1] = int(np.round(dim_arr[1]))
    else:
        dim_arr = np.array(dim)

    dim_arr = np.array(dim_arr, dtype=int)

    # When a local dimension is small, block positivity is trivial.
    if min(dim_arr) <= k:
        return is_positive_semidefinite(mat)

    op_norm = np.linalg.norm(mat, ord=2)
    # We compute the S(k)-norm of this operator since
    # X k-block positive iff:
    #   c >= S(k)-norm of(c*I - X)
    # See Corollary 4.2.9. of `[@Johnston_2012_Norms].
    c_mat = op_norm * np.eye(dim_xy) - mat
    lower_bound, upper_bound = sk_operator_norm(c_mat, k, dim_arr, op_norm, effort)

    # block positive
    # Note that QETLAB is more conservative here and multiplies
    # by (1 - rtol). After some experiments though, I found out
    # that probably due to numerical inaccuracies of CVXPY the check
    #     upper_bound <= op_norm * (1 - rtol)
    # would fail even for k - block positive matrices. So, we choose to
    # relax this inequality by increasing RHS. Additionally, the check
    #     upper_bound <= op_norm * (1 - rtol)
    # has the "undesired" property that increasing tolerance makes the
    # inequality more difficult to satisfy but usually the reverse holds,
    # i.e increased tolerance parameter relaxes the problem.
    if upper_bound <= op_norm * (1 + rtol):
        return True
    # not block positive
    if lower_bound >= op_norm * (1 - rtol):
        return False

    return RuntimeError(
        "Unable to determine k-block positivity. Please consider increasing the relative tolerance or the effort level."
    )
