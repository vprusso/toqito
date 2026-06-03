"""Check if circulant set of states is antidistinguishable."""

from typing import Any, Literal, cast

import numpy as np

from toqito.matrix_ops import vectors_to_gram_matrix
from toqito.matrix_props import is_circulant

# The full import path was specified here because the doctest workflow was failing when the above function could not be
# imported https://github.com/vprusso/toqito/issues/473


def is_antidistinguishable_circulant(
    states: list[np.ndarray[tuple[int, Literal[1]], np.dtype[np.inexact[Any]]]]
    | np.ndarray[tuple[int, int], np.dtype[np.inexact[Any]]],
    skip_circulant_check: bool = False,
) -> bool | np.bool_:
    r"""Check whether a circulant set of vectors is antidistinguishable or not.

    For more information, see [@johnston2025tight].

    The ability to determine whether a set of quantum states is antidistinguishable can be obtained, in the case where
    their Gram matrix is circulant, by a criterion on the eigenvalues of said matrix. More precisely, a set of circulant
    states is antidistinguishable if:

    \[
    \sqrt{\lambda_0}\leqslant\sum_{j=1}^{n-1}\sqrt{\lambda_j}
    \]

    with \(\lambda_0\geqslant\lambda_1\geqlant\cdots\geqslant\lambda_{n-1}\) being the eigenvalues of the Gram matrix.

    Args:
        states: A set of vectors consisting of quantum states to determine the antidistinguishability of, or their Gram
            matrix.
        skip_circulant_check: Whether to check if the provided set of states or Gram matrix is circulant. Raises
            a `ValueError` if not.`

    Returns:
        `True` if the vectors are antidistinguishable; `False` otherwise.

    Examples:
        The trine states are a well-known example of antidistinguishable states. They are defined as:

        \[
        u_1 = |0\rangle, \quad
        u_1 = -\frac{1}{2}\left(|0\rangle + \sqrt{3}|1\rangle\right), \quad \text{and} \quad
        u_2 = -\frac{1}{2}\left(|0\rangle - \sqrt{3}|1\rangle\right).
        \]

        The Gram matrix of these states is:

        \[
        \begin{pmatrix}
        1&-\frac12&-\frac12\\
        -\frac12&1&-\frac12\\
        -\frac12&-\frac12&1
        \end{pmatrix}
        \]

        with eigenvalues \(\frac32\geqslant\frac32\geqslant0\). In particular, since it is circulant, we can check
        that this set of states is antidistinguishable via the inequality

        \[
        \sqrt{\frac32}\leqslant\sqrt{\frac13}+\sqrt{0}.
        \]

        It can be checked in `toqito` that the trine states are antidistinguishable:

        ```python exec="1" source="above" result="text"
        from toqito.states import trine
        from toqito.state_props import is_antidistinguishable_circulant
        print(is_antidistinguishable_circulant(trine()))
        ```

    """
    if isinstance(states, list):  # We're given a list of states
        return is_antidistinguishable_circulant(
            vectors_to_gram_matrix(cast(list[np.ndarray[tuple[int, Literal[1]], np.dtype[np.inexact[Any]]]], states))
        )

    if not skip_circulant_check and not is_circulant(states):
        raise ValueError("The Gram matrix is not circulant.")

    sorted_eigvals = np.sort(np.real(np.fft.ifft(states[0])))[::-1]
    first_sqrt_eigval, *other_sqrt_eigvals = np.sqrt(np.maximum(sorted_eigvals, 0.0))

    lhs = first_sqrt_eigval
    rhs = np.sum(other_sqrt_eigvals)

    return lhs <= rhs or np.isclose(lhs, rhs)
