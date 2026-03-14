"""Permute systems is used to permute subsystems within a quantum state or an operator."""

import functools
import operator

import numpy as np
from scipy import sparse

from toqito.perms import vec


def permute_systems(
    input_mat: np.ndarray,
    perm: np.ndarray | list[int],
    dim: np.ndarray | list[int] | None = None,
    row_only: bool = False,
    inv_perm: bool = False,
) -> np.ndarray:
    r"""Permute subsystems within a state or operator.

    Permutes the order of the subsystems of the vector or matrix `input_mat` according to the permutation vector
    `perm`, where the dimensions of the subsystems are given by the vector `dim`. If `input_mat` is
    non-square and not a vector, different row and column dimensions can be specified by putting the row dimensions in
    the first row of `dim` and the columns dimensions in the second row of `dim`.

    If `row_only = True`, then only the rows of `input_mat` are permuted, but not the columns -- this is
    equivalent to multiplying `input_mat` on the left by the corresponding permutation operator, but not on the
    right.

    If `row_only = False`, then `dim` only needs to contain the row dimensions of the subsystems, even if
    `input_mat` is not square. If `inv_perm = True`, then the inverse permutation of `perm` is applied
    instead of `perm` itself.

    Examples:
        For spaces \(\mathcal{A}\) and \(\mathcal{B}\) where \(\text{dim}(\mathcal{A}) =
        \text{dim}(\mathcal{B}) = 2\) we may consider an operator \(X \in \mathcal{A} \otimes \mathcal{B}\).
        Applying the
        `permute_systems` function with vector \([1,0]\) on \(X\), we may reorient the spaces such that \(X \in
        \mathcal{B} \otimes \mathcal{A}\).

        For example, if we define \(X \in \mathcal{A} \otimes \mathcal{B}\) as

        \[
            X = \begin{pmatrix}
                1 & 2 & 3 & 4 \\
                5 & 6 & 7 & 8 \\
                9 & 10 & 11 & 12 \\
                13 & 14 & 15 & 16
            \end{pmatrix},
        \]

        then applying the `permute_systems` function on \(X\) to obtain \(X \in \mathcal{B} \otimes \mathcal{A}\)
        yield the following matrix

        \[
            X_{[1,0]} = \begin{pmatrix}
                1 & 3 & 2 & 4 \\
                9 & 11 & 10 & 12 \\
                5 & 7 & 6 & 8 \\
                13 & 15 & 14 & 16
            \end{pmatrix}.
        \]

        ```python exec="1" source="above"
        import numpy as np
        from toqito.perms import permute_systems

        test_input_mat = np.arange(1, 17).reshape(4, 4)

        print(permute_systems(test_input_mat, [1, 0]))
        ```


        For spaces \(\mathcal{A}, \mathcal{B}\), and \(\mathcal{C}\) where \(\text{dim}(\mathcal{A}) =
        \text{dim}(\mathcal{B}) = \text{dim}(\mathcal{C}) = 2\) we may consider an operator \(X \in \mathcal{A} \otimes
        \mathcal{B} \otimes \mathcal{C}\). Applying the `permute_systems` function with vector \([1,2,0]\) on
        \(X\), we may reorient the spaces such that \(X \in \mathcal{B} \otimes \mathcal{C} \otimes \mathcal{A}\).

        For example, if we define \(X \in \mathcal{A} \otimes \mathcal{B} \otimes \mathcal{C}\) as

        \[
            X =
            \begin{pmatrix}
                1 & 2 & 3 & 4, 5 & 6 & 7 & 8 \\
                9 & 10 & 11 & 12 & 13 & 14 & 15 & 16 \\
                17 & 18 & 19 & 20 & 21 & 22 & 23 & 24 \\
                25 & 26 & 27 & 28 & 29 & 30 & 31 & 32 \\
                33 & 34 & 35 & 36 & 37 & 38 & 39 & 40 \\
                41 & 42 & 43 & 44 & 45 & 46 & 47 & 48 \\
                49 & 50 & 51 & 52 & 53 & 54 & 55 & 56 \\
                57 & 58 & 59 & 60 & 61 & 62 & 63 & 64
            \end{pmatrix},
        \]

        then applying the `permute_systems` function on \(X\) to obtain \(X \in \mathcal{B} \otimes \mathcal{C}
        \otimes \mathcal{C}\) yield the following matrix

        \[
            X_{[1, 2, 0]} =
            \begin{pmatrix}
                1 & 5 & 2 & 6 & 3 & 7 & 4, 8 \\
                33 & 37 & 34 & 38 & 35 & 39 & 36 & 40 \\
                9 & 13 & 10 & 14 & 11 & 15 & 12 & 16 \\
                41 & 45 & 42 & 46 & 43 & 47 & 44 & 48 \\
                17 & 21 & 18 & 22 & 19 & 23 & 20 & 24 \\
                49 & 53 & 50 & 54 & 51 & 55 & 52 & 56 \\
                25 & 29 & 26 & 30 & 27 & 31 & 28 & 32 \\
                57 & 61 & 58 & 62 & 59 & 63 & 60 & 64
            \end{pmatrix}.
        \]

        ```python exec="1" source="above"
        import numpy as np
        from toqito.perms import permute_systems

        test_input_mat = np.arange(1, 65).reshape(8, 8)

        print(permute_systems(test_input_mat, [1, 2, 0]))
        ```



    Raises:
        ValueError: If dimension does not match the number of subsystems.

    Args:
        input_mat: The vector or matrix.
        perm: A permutation vector.
        dim: The default has all subsystems of equal dimension.
        row_only: Default: `False`
        inv_perm: Default: `True`

    Returns:
        The matrix or vector that has been permuted.

    """
    if len(input_mat.shape) == 1:
        input_mat_dims = (1, input_mat.shape[0])
    else:
        input_mat_dims = input_mat.shape

    is_vec = np.min(input_mat_dims) == 1
    num_sys = len(perm)

    if dim is None:
        x_tmp = input_mat_dims[0] ** (1 / num_sys) * np.ones(num_sys)
        y_tmp = input_mat_dims[1] ** (1 / num_sys) * np.ones(num_sys)
        dim_arr = np.array([x_tmp, y_tmp])
    elif isinstance(dim, list):
        dim_arr = np.array(dim)
    else:
        dim_arr = dim

    if is_vec:
        # 1 if column vector
        if len(input_mat.shape) > 1:
            vec_orien = 0
        # 2 if row vector
        else:
            vec_orien = 1

    if len(dim_arr.shape) == 1:
        # Force dim to be a row vector.
        dim_tmp = dim_arr[:].T
        if is_vec:
            dim_arr = np.ones((2, len(dim_arr)))
            dim_arr[vec_orien, :] = dim_tmp
        else:
            dim_arr = np.array([[dim_tmp], [dim_tmp]])

    prod_dim_r = int(np.prod(dim_arr[0, :]))
    prod_dim_c = int(np.prod(dim_arr[1, :]))

    if sorted(perm) != list(range(num_sys)):
        raise ValueError("InvalidPerm: `perm` must be a permutation vector.")
    if input_mat_dims[0] != prod_dim_r or (not row_only and input_mat_dims[1] != prod_dim_c):
        raise ValueError("InvalidDim: The dimensions specified in DIM do not agree with the size of X.")
    if is_vec:
        # If `input_mat` is a 1-by-X row vector, ensure we "flatten it" appropriately:
        if input_mat.shape[0] == 1:
            input_mat = input_mat[0]
            vec_orien = 1
        # Rather than using subtraction to generate new indices,
        # it's better to use methods designed for handling permutations directly.
        # This avoids the risk of negative indices and is more straightforward.
        num_sys -= 1  # 0-indexing (Since we're using 0-indexing, we need to subtract 1 from the number of subsystems.)
        permuted_mat_1 = input_mat.reshape(dim_arr[vec_orien, ::-1].astype(int), order="F")
        if inv_perm:
            permuted_mat = vec(np.transpose(permuted_mat_1, np.argsort(num_sys - np.array(perm[::-1])))).T
        else:
            permuted_mat = vec(np.transpose(permuted_mat_1, num_sys - np.array(perm[::-1]))).T

        # We need to flatten out the array.
        permuted_mat = functools.reduce(operator.iconcat, permuted_mat, [])
        return np.array(permuted_mat)

    vec_arg = np.array(list(range(0, input_mat_dims[0])))

    # If the dimensions are specified, ensure they are given to the
    # recursive calls as flattened lists.
    if len(dim_arr[0][:]) == 1:
        dim_arr = functools.reduce(operator.iconcat, dim_arr, [])

    row_perm = permute_systems(vec_arg, perm, dim_arr[0][:], False, inv_perm)

    # This condition is only necessary if the `input_mat` variable is sparse.
    if sparse.issparse(input_mat):
        input_mat = input_mat.toarray()
        permuted_mat = input_mat[row_perm, :]
        permuted_mat = np.array(permuted_mat)
    else:
        permuted_mat = input_mat[row_perm, :]

    if not row_only:
        vec_arg = np.array(list(range(0, input_mat_dims[1])))
        col_perm = permute_systems(vec_arg, perm, dim_arr[1][:], False, inv_perm)
        permuted_mat = permuted_mat[:, col_perm]

    return permuted_mat
