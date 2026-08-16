"""Tensor product operation calculates the tensor product between vectors or matrices."""

import numpy as np


def _kron(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Compute the Kronecker product of two arrays.

    This agrees with `numpy.kron` bitwise, but skips the per-call `expand_dims` and axis
    normalization that dominate the runtime for the small operands `|toqito⟩` tensors together.
    Inserting size-one axes with `reshape` is always a view, so non-contiguous operands are not
    copied.

    The `type(...) is not np.ndarray` test is deliberately stricter than `isinstance`. Subclasses
    such as `np.matrix` and masked arrays, along with scipy sparse operands, are handed to
    `numpy.kron` so that its `subok` wrapping and the sparse return types are preserved.
    """
    if type(mat_a) is not np.ndarray or type(mat_b) is not np.ndarray:
        return np.kron(mat_a, mat_b)

    if mat_a.ndim == 2 and mat_b.ndim == 2:
        rows_a, cols_a = mat_a.shape
        rows_b, cols_b = mat_b.shape
        return (mat_a.reshape(rows_a, 1, cols_a, 1) * mat_b.reshape(1, rows_b, 1, cols_b)).reshape(
            rows_a * rows_b, cols_a * cols_b
        )

    # `numpy.kron` pads the operand of lower dimension with leading singleton axes, so match that
    # rule before interleaving the axes of the two operands.
    ndim = max(mat_a.ndim, mat_b.ndim)
    shape_a = (1,) * (ndim - mat_a.ndim) + mat_a.shape
    shape_b = (1,) * (ndim - mat_b.ndim) + mat_b.shape
    view_a: list[int] = []
    view_b: list[int] = []
    shape_out: list[int] = []
    for dim_a, dim_b in zip(shape_a, shape_b, strict=True):
        view_a += [dim_a, 1]
        view_b += [1, dim_b]
        shape_out.append(dim_a * dim_b)
    return (mat_a.reshape(view_a) * mat_b.reshape(view_b)).reshape(shape_out)


def tensor(*args: np.ndarray | int | list[np.ndarray]) -> np.ndarray:
    r"""Compute the Kronecker tensor product [@wikipediatensor].

    Tensor two matrices or vectors together using the standard Kronecker
    operation provided from numpy.

    Given two matrices \(A\) and \(B\), computes \(A \otimes B\).
    The same concept also applies to two vectors \(v\) and \(w\) which
    computes \(v \otimes w\).

    One may also compute the tensor product one matrix \(n\) times with itself.

    For a matrix, \(A\) and an integer \(n\), the result of this
    function computes \(A^{\otimes n}\).

    Similarly for a vector \(v\) and an integer \(n\), the result of
    of this function computes \(v^{\otimes n}\).

    One may also perform the tensor product on a list of matrices.

    Given a list of \(n\) matrices \(A_1, A_2, \ldots, A_n\) the result
    of this function computes

    \[
        A_1 \otimes A_2 \otimes \cdots \otimes A_n.
    \]

    Similarly, for a list of \(n\) vectors \(v_1, v_2, \ldots, v_n\),
    the result of this function computes

    \[
        v_1 \otimes v_2 \otimes \cdots \otimes v_n.
    \]

    Args:
        args: Input to the tensor function is expected to be either:
            - list[np.ndarray]: List of numpy matrices,
            - np.ndarray, ... , np.ndarray: An arbitrary number of numpy arrays,
            - np.ndarray, int: A numpy array and an integer.

    Returns:
        The computed tensor product.

    Raises:
        ValueError: Input must be a vector or matrix.

    Examples:
        Tensor product two matrices or vectors

        Consider the following ket vector

        \[
            e_0 = \left[1, 0 \right]^{\text{T}}.
        \]

        Computing the following tensor product

        \[
        e_0 \otimes e_0 = [1, 0, 0, 0]^{\text{T}}.
        \]

        This can be accomplished in `|toqito⟩` as follows.
        ```python exec="1" source="above" result="text"
        from toqito.states import basis
        from toqito.matrix_ops import tensor

        e_0 = basis(2, 0)

        print(tensor(e_0, e_0))
        ```

        Tensor product one matrix \(n\) times with itself.

        We may also tensor some element with itself some integer number of times.
        For instance we can compute

        \[
            e_0^{\otimes 3} = \left[1, 0, 0, 0, 0, 0, 0, 0 \right]^{\text{T}}
        \]

        in `|toqito⟩` as follows.
        ```python exec="1" source="above" result="text"
        from toqito.states import basis
        from toqito.matrix_ops import tensor

        e_0 = basis(2, 0)

        print(tensor(e_0, 3))
        ```

        Perform the tensor product on a list of vectors or matrices.

        If we wish to compute the tensor product against more than two matrices or
        vectors, we can feed them in as a `list`. For instance, if we wish to
        compute \(e_0 \otimes e_1 \otimes e_0\), we can do
        so as follows.
        ```python exec="1" source="above" result="text"
        from toqito.states import basis
        from toqito.matrix_ops import tensor

        e_0, e_1 = basis(2, 0), basis(2, 1)

        print(tensor([e_0, e_1, e_0]))
        ```

    """

    def fast_exp(matrix: np.ndarray, q: int) -> np.ndarray:
        """Efficient exponentiation by squaring."""
        if q == 1:
            return matrix
        tmp = fast_exp(matrix, q >> 1)
        tmp = _kron(tmp, tmp)
        if q & 1:  # If q is odd
            tmp = _kron(matrix, tmp)
        return tmp

    result = None

    # Input is provided as a list of numpy matrices.
    if (len(args) == 1 and isinstance(args[0], list)) or (len(args) == 1 and isinstance(args[0], np.ndarray)):
        if len(args[0]) == 0:
            raise ValueError("The `tensor` function requires at least one matrix; the input list is empty.")
        if len(args[0]) == 1:
            return args[0][0]
        if len(args[0]) == 2:
            return _kron(args[0][0], args[0][1])
        result = args[0][0]
        for i in range(1, len(args[0])):
            result = _kron(result, args[0][i])
        return result

    # Tensor product one matrix `n` times with itself.
    if len(args) == 2 and isinstance(args[1], int):
        num_tensor = args[1]
        if num_tensor == 0:
            return np.eye(1, dtype=args[0].dtype)
        if num_tensor == 1:
            return args[0]
        return fast_exp(args[0], num_tensor)

    # Tensor product between two or more matrices.
    if len(args) == 2:
        return _kron(args[0], args[1])
    if len(args) >= 3:
        result = args[0]
        for i in range(1, len(args)):
            result = _kron(result, args[i])
        return result

    raise ValueError("The `tensor` function must take either a matrix or vector.")
