"""Calculates the permutation sign."""

import numpy as np
from scipy import linalg


def perm_sign(perm: np.ndarray | list[int]) -> float:
    r"""Compute the "sign" of a permutation [@WikiParPerm].

    The sign (either -1 or 1) of the permutation `perm` is `-1**inv`, where `inv` is the number of
    inversions contained in `perm`.

    Examples:
        For the following vector

        \[
            [1, 2, 3, 4]
        \]

        the permutation sign is positive as the number of elements in the vector are even. This can be performed in
        `|toqito⟩` as follows.

        ```python exec="1" source="above"
        from toqito.perms import perm_sign

        print(perm_sign([1, 2, 3, 4]))
        ```

        For the following vector

        \[
            [1, 2, 3, 4, 5]
        \]

        the permutation sign is negative as the number of elements in the vector are odd. This can be performed in
        `|toqito⟩` as follows.

        ```python exec="1" source="above"
        from toqito.perms import perm_sign

        print(perm_sign([1, 2, 4, 3, 5]))
        ```

    Args:
        perm: The permutation vector to be checked.

    Returns:
        The value 1 if the permutation is of even length and the value of -1 if the permutation is of odd length.

    """
    return linalg.det(np.eye(len(perm))[:, np.array(perm) - 1])
