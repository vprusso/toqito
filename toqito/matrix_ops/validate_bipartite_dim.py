"""Validates and normalizes a bipartite subsystem dimension specification."""

import numpy as np


def validate_bipartite_dim(
    rho: np.ndarray, dim: int | list[int] | tuple | np.ndarray | None
) -> np.ndarray:
    r"""Validate and normalize a bipartite dimension specification.

    Given a square operator ``rho`` on a tensor-product space and a ``dim``
    argument describing how that space splits into two subsystems, return the
    two subsystem dimensions as a length-two integer array.

    If ``dim`` is ``None`` the two subsystems are assumed equal, which requires
    the dimension of ``rho`` to be a perfect square. A scalar ``dim`` is taken as
    the first subsystem dimension and must divide the dimension of ``rho``.

    Args:
        rho: A square operator on the bipartite space.
        dim: The subsystem dimensions. Either ``None`` (assume equal
            subsystems), a scalar (the first subsystem dimension), or a pair of
            dimensions whose product matches the dimension of ``rho``.

    Returns:
        A length-two integer array of subsystem dimensions.

    Raises:
        ValueError: If the dimensions cannot be inferred, are not integers, are
            not positive, or do not multiply to the dimension of ``rho``.

    Examples:
        With equal subsystems the dimensions are inferred from the size of
        ``rho``:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_ops import validate_bipartite_dim

        print(validate_bipartite_dim(np.eye(6), [2, 3]))
        ```

    """
    n = rho.shape[0]

    if dim is None:
        d = int(round(np.sqrt(n)))
        if d * d != n:
            raise ValueError("Cannot infer bipartite subsystem dimensions directly. Please provide `dim`.")
        return np.array([d, d], dtype=int)

    if isinstance(dim, int):
        if dim <= 0 or n % dim != 0:
            raise ValueError("If `dim` is a scalar, it must be a positive divisor of the matrix dimension.")
        return np.array([dim, n // dim], dtype=int)

    dims = np.asarray(dim)
    if dims.ndim != 1 or len(dims) != 2:
        raise ValueError("`dim` must describe exactly two subsystem dimensions.")
    if not np.issubdtype(dims.dtype, np.integer):
        raise ValueError("`dim` must contain integer subsystem dimensions.")
    if np.any(dims <= 0):
        raise ValueError("Subsystem dimensions in `dim` must be positive.")
    if int(np.prod(dims)) != n:
        raise ValueError("The product of `dim` must match the dimension of `rho`.")
    return dims.astype(int)
