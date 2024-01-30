"""Is matrix block positive."""


import numpy as np

from toqito.matrix_props.is_hermitian import is_hermitian
from toqito.matrix_props.is_positive_semidefinite import is_positive_semidefinite
from toqito.matrix_props.sk_norm import sk_operator_norm


def is_block_positive(
    mat: np.ndarray,
    k: int = 1,
    dim: int | list[int] = None,
    effort: int = 2,
    rtol: float = 1e-5,
) -> bool:
    r"""Check if matrix is block positive :cite:`Johnston_2012_Norms`.

    Examples
    ==========

    The swap operator is always block positive, since it is the Choi
    matrix of the transpose map.

    >>> from toqito.matrix_props import is_block_positive
    >>> from toqito.perms import swap_operator
    >>>
    >>> mat = swap_operator(3)
    >>> is_block_positive(mat)
    True

    However, it's not 2 - block positive.

    >>> from toqito.matrix_props import is_block_positive
    >>> from toqito.perms import swap_operator
    >>>
    >>> mat = swap_operator(3)
    >>> is_block_positive(mat, k=2)
    False

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises RuntimeError: Unable to determine k-block positivity. Please consider increasing the relative tolerance or
                            the effort level.
    :param mat: A bipartite Hermitian operator.
    :param k: A positive integer indicating that the function should determine whether or not
              the input operator is k-block positive, i.e., whether or not it remains nonnegative
              under left and right multiplication by vectors with Schmidt rank <= k (default 1).
    :param dim: The dimension of the two sub-systems. By default it's assumed to be equal.
    :param effort: An integer value indicating the amount of computation you want to devote to
                   determine block positivity before giving up.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :return: Return :code:`True` if matrix is k-block positive definite,
             :code:`False` if not, or raise a runtime error if we are unable to determine
             whether or not the operator is block positive.

    """
    if not is_hermitian(mat):
        return False

    dim_xy = mat.shape[0]
    # Set default dimension if none was provided.
    if dim is None:
        dim = int(np.round(np.sqrt(dim_xy)))

    # Allow the user to enter in a single integer for dimension.
    if isinstance(dim, int):
        dim = np.array([dim, dim_xy / dim])  # pylint: disable=redefined-variable-type
        dim[1] = int(np.round(dim[1]))

    dim = np.array(dim, dtype=int)

    # When a local dimension is small, block positivity is trivial.
    if min(dim) <= k:
        return is_positive_semidefinite(mat)

    op_norm = np.linalg.norm(mat, ord=2)
    # We compute the S(k)-norm of this operator since
    # X k-block positive iff:
    #   c >= S(k)-norm of(c*I - X)
    # See Corollary 4.2.9. of `:cite:`Johnston_2012_Norms`.
    c_mat = op_norm * np.eye(dim_xy) - mat
    lower_bound, upper_bound = sk_operator_norm(c_mat, k, dim, op_norm, effort)

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
