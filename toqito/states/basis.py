"""A basis state represents the standard basis vectors of some n-dimensional Hilbert Space.

Here, n can be given as a parameter as shown below.
"""

import numpy as np


def basis(dim: int, pos: int) -> np.ndarray:
    r"""Obtain the ket of dimension `dim` [@WikiBraKet].

    Examples:
    The standard basis ket vectors given as \(|0 \rangle\) and \(|1 \rangle\) where

    \[
        |0 \rangle = \left[1, 0 \right]^{\text{T}} \quad \text{and} \quad
        |1 \rangle = \left[0, 1 \right]^{\text{T}},
    \]

    can be obtained in `|toqito⟩` as follows.

    Example:  Ket basis vector: \(|0\rangle\).

    ```python exec="1" source="above"
    from toqito.states import basis
    print(basis(2, 0))
    ```

    Example: Ket basis vector: \(|1\rangle\).

    ```python exec="1" source="above"
    from toqito.states import basis
    print(basis(2, 1))
    ```

    Raises:
        ValueError: If the input position is not in the range [0, dim - 1].

    Args:
        dim: The dimension of the column vector.
        pos: 0-indexed position of the basis vector where the 1 will be placed.

    Returns:
        The column vector of dimension `dim` with all entries set to `0` except the entry at `pos` which is set to `1`.

    """
    if pos >= dim or pos < 0:
        raise ValueError("Invalid: The `pos` variable needs to be between [0, dim - 1] for ket function.")

    ret = np.zeros(dim, dtype=np.int64)
    ret[pos] = 1
    return ret.reshape(-1, 1)
