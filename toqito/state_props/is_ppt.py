"""Checks if a quantum state violates the PPT criterion."""

import numpy as np

from toqito.matrix_ops import partial_transpose
from toqito.matrix_props import is_positive_semidefinite


def is_ppt(
    mat: np.ndarray, sys: int = 2, dim: int | list[int] | np.ndarray | None = None, tol: float | None = None
) -> bool:
    r"""Determine whether or not a matrix has positive partial transpose [@WikiPeresHorodecki].

    Yields either `True` or `False`, indicating that `mat` does or does not have
    positive partial transpose (within numerical error). The variable `mat` is assumed to act
    on bipartite space.

    For shared systems of \(2 \otimes 2\) or \(2 \otimes 3\), the PPT criterion serves as a
    method to determine whether a given state is entangled or separable. Therefore, for systems of
    this size, the return value `True` would indicate that the state is separable and a value
    of `False` would indicate the state is entangled.

    Examples:
        Consider the following matrix

        \[
            X =
            \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
            \end{pmatrix}.
        \]

        This matrix trivially satisfies the PPT criterion as can be seen using the
        `|toqito⟩` package.

        ```python exec="1" source="above"
        from toqito.state_props import is_ppt
        import numpy as np
        mat = np.identity(9)
        print(is_ppt(mat))
        ```

        Consider the following Bell state:

        \[
            u = \frac{1}{\sqrt{2}}\left( |01 \rangle + |10 \rangle \right).
        \]

        For the density matrix \(\rho = u u^*\), as this is an entangled state
        of dimension \(2\), it will violate the PPT criterion, which can be seen
        using the `|toqito⟩` package.

        ```python exec="1" source="above"
        from toqito.states import bell
        from toqito.state_props import is_ppt
        rho = bell(2) @ bell(2).conj().T
        print(is_ppt(rho))
        ```

    Args:
        mat: A square matrix.
        sys: Scalar or vector indicating which subsystems the transpose should be applied on.
        dim: The dimension is a vector containing the dimensions of the subsystems on which `mat` acts.
        tol: Tolerance with which to check whether `mat` is PPT.

    Returns:
        Returns `True` if `mat` is PPT and `False` if not.

    """
    eps = np.finfo(float).eps

    sqrt_rho_dims = np.round(np.sqrt(list(mat.shape)))
    sqrt_rho_dims = np.int_(sqrt_rho_dims)

    if dim is None:
        dim = [
            [sqrt_rho_dims[0], sqrt_rho_dims[0]],
            [sqrt_rho_dims[1], sqrt_rho_dims[1]],
        ]
    if tol is None:
        tol = np.sqrt(eps)
    return is_positive_semidefinite(partial_transpose(mat, [sys - 1], dim), tol)
