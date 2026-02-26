"""Checks if the quantum state has NPT (negative partial transpose) criterion."""

import numpy as np

from toqito.state_props import is_ppt


def is_npt(mat: np.ndarray, sys: int = 2, dim: int | list[int] | None = None, tol: float | None = None) -> bool:
    r"""Determine whether or not a matrix has negative partial transpose [@WikiPeresHorodecki].

    Yields either `True` or `False`, indicating that `mat` does or does not have
    negative partial transpose (within numerical error). The variable `mat` is assumed to act
    on bipartite space. [@DiVincenzo_2000_Evidence]

    A state has negative partial transpose if it does not have positive partial transpose.

    Examples:
    To check if a matrix has negative partial transpose

    ```python exec="1" source="above"
    import numpy as np
    from toqito.state_props import is_npt
    from toqito.states import bell
    print(is_npt(bell(2) @ bell(2).conj().T, 2))
    ```

    Args:
        mat: A square matrix.
        sys: Scalar or vector indicating which subsystems the transpose should be applied on. Default value is `2`.
        dim: The dimension is a vector containing the dimensions of the subsystems on which `mat` acts.
        tol: Tolerance with which to check whether `mat` is PPT.

    Returns:
        Returns `True` if `mat` is NPT and `False` if not.

    """
    return not is_ppt(mat, sys, dim, tol)
