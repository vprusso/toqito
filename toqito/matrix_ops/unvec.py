"""Unvec operation is used to perform the unvec operation on a vector."""

import numpy as np


def unvec(vector: np.ndarray, shape: list[int] | None = None) -> np.ndarray:
    r"""Perform the unvec operation on a vector to obtain a matrix [@Rigetti_2022_Forest].

    Takes a column vector and transforms it into a `shape[0]`-by-`shape[1]` matrix.
    This operation is the inverse of `vec` operation in `|toqito⟩`.

    For instance, for the following column vector

    \[
        u = \begin{pmatrix} 1 \\ 3 \\ 2 \\ 4 \end{pmatrix},
    \]

    it holds that

    \[
        \text{unvec}(u) =
        \begin{pmatrix}
            1 & 2 \\
            3 & 4
        \end{pmatrix}
    \]

    More formally, the vec operation is defined by

    \[
        \text{unvec}(e_a \otimes e_b) = E_{a,b}
    \]

    for all \(a\) and \(b\) where

    \[
        E_{a,b}(c,d) = \begin{cases}
                          1 & \text{if} \ (c,d) = (a,b) \\
                          0 & \text{otherwise}
                        \end{cases}
    \]

    for all \(c\) and \(d\) and where

    \[
        e_a(b) = \begin{cases}
                     1 & \text{if} \ a = b \\
                     0 & \text{if} \ a \not= b
                 \end{cases}
    \]

    for all \(a\) and \(b\).

    This function has been adapted from [@Rigetti_2022_Forest].

    Examples:
        Consider the following vector

        \[
            u = \begin{pmatrix} 1 \\ 3 \\ 2 \\ 4 \end{pmatrix}
        \]

        Performing the \(\text{unvec}\) operation on \(u\) yields

        \[
            \text{unvec}(u) = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
        \]

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_ops import unvec

        u = np.array([1, 2, 3, 4])

        print(unvec(u))
        ```

        !!! See Also
            [vec][toqito.matrix_ops.vec.vec]

    Args:
        vector: A (`shape[0] * shape[1]`)-by-1 numpy array.
        shape: The shape of the output matrix; by default, the matrix is assumed to be square.

    Returns:
        Returns a `shape[0]`-by-`shape[1]` matrix.

    """
    vector = np.asarray(vector)
    if shape is None:
        dim = int(np.sqrt(vector.size))
        shape = dim, dim
    mat = vector.reshape(*shape, order="F")
    return mat
