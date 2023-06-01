"""The partial trace."""
from __future__ import annotations

import numpy as np
import picos as pc

from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from toqito.perms import permute_systems
from toqito.helper import expr_as_np_array, np_array_as_expr


def partial_trace(
    input_mat: np.ndarray | Variable,
    sys: list[int] = [1],
    dim: list[int] = [2, 2],
) -> np.ndarray | Expression:
    r"""
    Compute the partial trace of a matrix [WikPtrace]_.

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
                 \end{pmatrix}

    By default, the partial trace function in :code:`toqito` takes the trace of the second
    subsystem.

    >>> from toqito.channels import partial_trace
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> partial_trace(test_input_mat)
    [[ 7, 11],
     [23, 27]]

    By specifying the :code:`sys = 1` argument, we can perform the partial trace over the first
    subsystem (instead of the default second subsystem as done above). Performing the partial
    trace over the first subsystem yields the following matrix

    .. math::
        X_{pt, 1} = \begin{pmatrix}
                        12 & 14 \\
                        20 & 22
                    \end{pmatrix}

    >>> from toqito.channels import partial_trace
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> partial_trace(test_input_mat, [0])
    [[12, 14],
     [20, 22]]

    We can also specify both dimension and system size as :code:`list` arguments. Consider the
    following :math:`16`-by-:math:`16` matrix.

    >>> from toqito.channels import partial_trace
    >>> import numpy as np
    >>> test_input_mat = np.arange(1, 257).reshape(16, 16)
    >>> test_input_mat
    [[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16]
     [ 17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32]
     [ 33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48]
     [ 49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64]
     [ 65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80]
     [ 81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96]
     [ 97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112]
     [113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128]
     [129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144]
     [145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160]
     [161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176]
     [177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192]
     [193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208]
     [209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224]
     [225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240]
     [241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256]]

    We can take the partial trace on the first and third subsystems and assume that the size of
    each of the 4 systems is of dimension 2.

    >>> from toqito.channels import partial_trace
    >>> import numpy as np
    >>> partial_trace(test_input_mat, [0, 2], [2, 2, 2, 2])
    [ 3.44e+02  3.48e+02  3.60e+02  3.64e+02]
    [ 4.08e+02  4.12e+02  4.24e+02  4.28e+02]
    [ 6.00e+02  6.04e+02  6.16e+02  6.20e+02]
    [ 6.64e+02  6.68e+02  6.80e+02  6.84e+02]

    References
    ==========
    .. [WikPtrace] Wikipedia: Partial trace
        https://en.wikipedia.org/wiki/Partial_trace

    :raises ValueError: If matrix dimension is not equal to the number of subsystems.
    :param input_mat: A square matrix.
    :param sys: Vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If :code:`None`, all dimensions are assumed to be
                equal.
    :return: The partial trace of matrix :code:`input_mat`.
    """
    # If the input matrix is a CVX variable for an SDP, we convert it to a numpy array,
    # perform the partial trace, and convert it back to a CVX variable.
    if isinstance(input_mat, Variable):
        if isinstance(sys, (list, np.ndarray)):
            if len(sys) == 1:
                pt_mat = partial_trace_for_cvxpy_variable(input_mat, sys[0]+1, dim)
                return pt_mat
            else:
                pt_mat = partial_trace_for_cvxpy_variable(input_mat, np.array(sys)+1, dim)
                return pt_mat
        else:
            pt_mat = partial_trace_for_cvxpy_variable(input_mat, sys+1, dim)
    
    pt_mat = pc.partial_trace(input_mat, sys, dim)
    return pt_mat
    
def partial_trace_for_cvxpy_variable(
    input_mat: np.ndarray | Variable,
    sys: int | list[int] = 2,
    dim: int | list[int] = None,
) -> np.ndarray | Expression:
    
    input_mat = expr_as_np_array(input_mat)

    if dim is None:
        dim = np.array([np.round(np.sqrt(len(input_mat)))])
    if isinstance(dim, int):
        dim = np.array([dim])
    if isinstance(dim, list):
        dim = np.array(dim)

    num_sys = len(dim)

    # Allow the user to enter a single number for dim.
    if num_sys == 1:
        dim = np.array([dim[0], len(input_mat) / dim[0]])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * len(input_mat) * np.finfo(float).eps:
            raise ValueError(
                "Invalid: If `dim` is a scalar, `dim` must evenly divide `len(input_mat)`."
            )
        dim[1] = np.round(dim[1])
        num_sys = 2

    prod_dim = np.prod(dim)
    if isinstance(sys, (list, np.ndarray)):
        if len(sys) == 1:
            prod_dim_sys = np.prod(dim[sys[0] - 1])
        else:
            prod_dim_sys = 1
            for idx in sys:
                prod_dim_sys *= dim[idx - 1]
    elif isinstance(sys, int):
        prod_dim_sys = np.prod(dim[sys - 1])
    else:
        raise ValueError(
            "Invalid: The variable `sys` must either be of type int or of a list of ints."
        )

    sub_prod = prod_dim / prod_dim_sys
    sub_sys_vec = prod_dim * np.ones(int(sub_prod)) / sub_prod

    if isinstance(sys, int):
        sys = [sys]
    set_diff = list(set(list(range(1, num_sys + 1))) - set(sys))

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

    pt_mat = permuted_reshaped_mat[
        :, :, list(range(0, int(sub_sys_vec[0] ** 2), int(sub_sys_vec[0] + 1)))
    ]
    pt_mat = np.sum(pt_mat, axis=2)
    traced_rho = np_array_as_expr(pt_mat)

    return traced_rho
