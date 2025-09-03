"""Generates the partial trace of a matrix."""

import numpy as np
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable

from toqito.helper import expr_as_np_array, np_array_as_expr
from toqito.perms import permute_systems


def partial_trace(
    input_mat: np.ndarray | Variable,
    sys: int | list[int] = None,
    dim: int | list[int] = None,
) -> np.ndarray | Expression:
    r"""Compute the partial trace of a matrix :footcite:`WikiPartialTr`.

    The *partial trace* is defined as

    .. math::
        \left( \text{Tr} \otimes \mathbb{I}_{\mathcal{Y}} \right)
        \left(X \otimes Y \right) = \text{Tr}(X)Y

    where :math:`X \in \text{L}(\mathcal{X})` and :math:`Y \in \text{L}(\mathcal{Y})` are linear
    operators over complex Euclidean spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}`.

    Gives the partial trace of the matrix X, where the dimensions of the (possibly more than 2)
    subsystems are given by the vector :code:`dim` and the subsystems to take the trace on are
    given by the scalar or vector :code:`sys`.

    Examples
    ==========

    Consider the following matrix

    .. math::
        X = \begin{pmatrix}
                1 & 2 & 3 & 4 \\
                5 & 6 & 7 & 8 \\
                9 & 10 & 11 & 12 \\
                13 & 14 & 15 & 16
            \end{pmatrix}.

    Taking the partial trace over the second subsystem of :math:`X` yields the following matrix

    .. math::
        X_{pt, 2} = \begin{pmatrix}
                    7 & 11 \\
                    23 & 27
                 \end{pmatrix}.

    By default, the partial trace function in :code:`|toqito⟩` takes the trace of the second
    subsystem.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_ops import partial_trace

     test_input_mat = np.arange(1, 17).reshape(4, 4)

     partial_trace(test_input_mat)


    By specifying the :code:`sys = [0]` argument, we can perform the partial trace over the first
    subsystem (instead of the default second subsystem as done above). Performing the partial
    trace over the first subsystem yields the following matrix

    .. math::
        X_{pt, 1} = \begin{pmatrix}
                        12 & 14 \\
                        20 & 22
                    \end{pmatrix}

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_ops import partial_trace

     test_input_mat = np.arange(1, 17).reshape(4, 4)

     partial_trace(test_input_mat, [0])

    We can also specify both dimension and system size as :code:`list` arguments. Consider the
    following :math:`16`-by-:math:`16` matrix.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_ops import partial_trace

     test_input_mat = np.arange(1, 257).reshape(16, 16)
     test_input_mat


    We can take the partial trace on the first and third subsystems and assume that the size of
    each of the 4 systems is of dimension 2.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_ops import partial_trace

     partial_trace(test_input_mat, [0, 2], [2, 2, 2, 2])


    References
    ==========
    .. footbibliography::



    :raises ValueError: If matrix dimension is not equal to the number of subsystems.
    :param input_mat: A square matrix.
    :param sys: Scalar or vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If :code:`None`, all dimensions are assumed to be
                equal.
    :return: The partial trace of matrix :code:`input_mat`.

    """
    if not isinstance(sys, int):
        if sys is None:
            sys = [1]
    # If the input matrix is a CVX variable for an SDP, we convert it to a numpy array,
    # perform the partial trace, and convert it back to a CVX variable.
    if isinstance(input_mat, Variable):
        rho_np = expr_as_np_array(input_mat)
        traced_rho = partial_trace(rho_np, sys, dim)
        traced_rho = np_array_as_expr(traced_rho)
        return traced_rho

    n = input_mat.shape[0]
    if dim is None:
        d = int(round(np.sqrt(n)))
        if d * d != n:
            raise ValueError("Cannot infer subsystem dimensions directly. Please provide `dim`.")
        dim = np.array([d, d])
    elif isinstance(dim, int):
        if n % dim != 0:
            raise ValueError("Invalid: If `dim` is a scalar, it must evenly divide matrix dimension.")
        dim = np.array([dim, n // dim])
    elif isinstance(dim, list):
        if len(dim) == 1:
            d = dim[0]
            if n % d != 0:
                raise ValueError("Invalid: If `dim` is a scalar, it must evenly divide matrix dimension.")
            dim = np.array([d, n // d])
        else:
            dim = np.array(dim)

    num_sys = len(dim)
    prod_dim = np.prod(dim)
    if isinstance(sys, int):
        prod_dim_sys = dim[sys]
        sys = np.array([sys])
    elif isinstance(sys, (list, np.ndarray)):
        prod_dim_sys = int(np.prod([dim[i] for i in sys]))
        sys = np.array(sys)
    else:
        raise ValueError("Invalid: The variable `sys` must either be of type int or of a list of ints.")

    sub_prod = prod_dim // prod_dim_sys
    sub_sys_size = prod_dim_sys

    remaining_sys = np.setdiff1d(np.arange(num_sys), sys, assume_unique=True)
    perm = np.concatenate([remaining_sys, sys]).astype(np.int32)

    a_mat = permute_systems(input_mat, perm, dim)

    ret_mat = np.reshape(
        a_mat,
        [sub_sys_size, sub_prod, sub_sys_size, sub_prod],
        order="F",
    )
    permuted_mat = ret_mat.transpose((1, 3, 0, 2))

    permuted_reshaped_mat = np.reshape(
        permuted_mat,
        [sub_prod, sub_prod, sub_sys_size**2],
        order="F",
    )

    diag_idx = np.arange(sub_sys_size) * (sub_sys_size + 1)
    pt_mat = permuted_reshaped_mat[:, :, diag_idx]
    pt_mat = np.sum(pt_mat, axis=2)

    return pt_mat
