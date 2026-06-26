"""Calculates the permutation sign."""

import numpy as np
from scipy import linalg


def perm_sign(perm: np.ndarray | list[int]) -> float:
    r"""Compute the "sign" of a permutation [@wikipediaparity].

    The sign (either -1 or 1) of the permutation `perm` is `(-1)**inv`, where `inv` is the number of
    inversions contained in `perm`.

    The permutation is expected to use 1-based labels, i.e. a permutation of `[1, 2, ..., n]`.

    Args:
        perm: The permutation vector (using 1-based labels `1, ..., n`) to be checked.

    Returns:
        The value 1 if the permutation is even (an even number of inversions) and -1 if it is odd.

    Examples:
        For the following vector

        \[
            [1, 2, 3, 4]
        \]

        the permutation sign is positive because it is the identity permutation (no inversions). This can be performed
        in `|toqito⟩` as follows.

        ```python exec="1" source="above" result="text"
        from toqito.perms import perm_sign

        print(perm_sign([1, 2, 3, 4]))
        ```

        For the following vector

        \[
            [1, 2, 4, 3, 5]
        \]

        the permutation sign is negative because it contains a single inversion (the pair 4, 3). This can be performed
        in `|toqito⟩` as follows.

        ```python exec="1" source="above" result="text"
        from toqito.perms import perm_sign

        print(perm_sign([1, 2, 4, 3, 5]))
        ```

    """
    return linalg.det(np.eye(len(perm))[:, np.array(perm) - 1])
