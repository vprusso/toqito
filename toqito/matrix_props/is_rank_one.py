"""Checks if a matrix has rank one."""

import warnings

import numpy as np


def is_rank_one(mat: np.ndarray, rtol: float = 1e-08, atol: float = 1e-08, *, tol: float | None = None) -> bool:
    r"""Determine whether the given matrix has rank one [@wikipediarank].

    The function evaluates the singular values (equivalently, eigenvalues for Hermitian matrices)
    and counts how many exceed the threshold ``rtol * max_singular_value`` (or ``atol`` when the
    matrix has no singular values).

    Args:
        mat: Matrix to test.
        rtol: Relative tolerance, applied to the largest singular value (default 1e-08).
        atol: Absolute tolerance, used as the threshold when there are no singular values
            (default 1e-08).
        tol: Deprecated alias retained for backward compatibility; if given it sets both ``rtol``
            and ``atol``.

    Returns:
        `True` if the matrix has rank at most one, `False` otherwise.

    Examples:
        Consider the Bell state density matrix \(\rho = \ket{\Phi^+}\bra{\Phi^+}\). This matrix
        has rank one.

        ```python exec="1" source="above" result="text"
        from toqito.matrix_props import is_rank_one
        from toqito.states import bell

        rho = bell(0) @ bell(0).conj().T
        print(is_rank_one(rho))
        ```

        On the other hand, the maximally mixed state is not rank one.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_rank_one

        maximally_mixed = np.eye(2) / 2
        print(is_rank_one(maximally_mixed))
        ```

    """
    if tol is not None:
        warnings.warn(
            "`tol` is deprecated; use `rtol` and `atol` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        rtol = atol = tol

    singular_values = np.linalg.svd(mat, compute_uv=False)
    # Threshold relative to the largest singular value so the result is independent of the matrix scale.
    threshold = rtol * singular_values.max() if singular_values.size else atol
    return bool(np.count_nonzero(singular_values > threshold) <= 1)
