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

    if dim is None:
        dim = np.array([np.round(np.sqrt(len(input_mat)))])
    if isinstance(dim, int):
        dim = np.array([dim])
    if isinstance(dim, list):
        dim = np.array(dim)

    # Allow the user to enter a single number for dim.
    if (num_sys := len(dim)) == 1:
        dim = np.array([dim[0], len(input_mat) / dim[0]])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * len(input_mat) * np.finfo(float).eps:
            raise ValueError("Invalid: If `dim` is a scalar, `dim` must evenly divide `len(input_mat)`.")
        dim[1] = np.round(dim[1])
        num_sys = 2

    prod_dim = np.prod(dim)
    if isinstance(sys, list):
        if len(sys) == 1:
            prod_dim_sys = np.prod(dim[sys[0]])
        else:
            prod_dim_sys = 1
            for idx in sys:
                prod_dim_sys *= dim[idx]
    elif isinstance(sys, int):
        prod_dim_sys = np.prod(dim[sys])
    else:
        raise ValueError("Invalid: The variable `sys` must either be of type int or of a list of ints.")

    sub_prod = prod_dim / prod_dim_sys
    sub_sys_vec = prod_dim * np.ones(int(sub_prod)) / sub_prod

    if isinstance(sys, list):
        sys = np.array(sys)
    if isinstance(sys, int):
        sys = np.array([sys])

    set_diff = list(set(list(range(num_sys))) - set(sys))
    perm = set_diff
    perm.extend(sys)

    a_mat = permute_systems(input_mat, perm, dim)

    ret_mat = np.reshape(
        a_mat,
        [int(sub_sys_vec[0]), int(sub_prod), int(sub_sys_vec[0]), int(sub_prod)],
        order="F",
    )
    permuted_mat = ret_mat.transpose((1, 3, 0, 2))

    permuted_reshaped_mat = np.reshape(
        permuted_mat,
        [int(sub_prod), int(sub_prod), int(sub_sys_vec[0] ** 2)],
        order="F",
    )

    pt_mat = permuted_reshaped_mat[:, :, list(range(0, int(sub_sys_vec[0] ** 2), int(sub_sys_vec[0] + 1)))]
    pt_mat = np.sum(pt_mat, axis=2)

    return pt_mat
