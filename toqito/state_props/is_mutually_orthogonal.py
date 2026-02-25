"""Checks if quantum states are mutually orthogonal."""

from typing import Any

import numpy as np


def is_mutually_orthogonal(vec_list: list[np.ndarray | list[float | Any]]) -> bool:
    r"""Check if list of vectors are mutually orthogonal [@WikiOrthog].

    We say that two bases

    \[
        \begin{equation}
            \mathcal{B}_0 = \left\{u_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
            \quad \text{and} \quad
            \mathcal{B}_1 = \left\{v_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
        \end{equation}
    \]

    are *mutually orthogonal* if and only if
    \(\left|\langle u_a, v_b \rangle\right| = 0\) for all \(a, b \in \Sigma\).

    For \(n \in \mathbb{N}\), a set of bases \(\left\{
    \mathcal{B}_0, \ldots, \mathcal{B}_{n-1} \right\}\) are mutually orthogonal if and only if
    every basis is orthogonal with every other basis in the set, i.e. \(\mathcal{B}_x\)
    is orthogonal with \(\mathcal{B}_x^{\prime}\) for all \(x \not= x^{\prime}\) with
    \(x, x^{\prime} \in \Sigma\).

    Examples:

    The Bell states constitute a set of mutually orthogonal vectors.

    ```python exec="1" source="above"
    from toqito.states import bell
    from toqito.state_props import is_mutually_orthogonal
    states = [bell(0), bell(1), bell(2), bell(3)]
    print(is_mutually_orthogonal(states))
    ```


    The following is an example of a list of vectors that are not mutually orthogonal.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.states import bell
    from toqito.state_props import is_mutually_orthogonal
    states = [np.array([1, 0]), np.array([1, 1])]
    print(is_mutually_orthogonal(states))
    ```

    Raises:
        ValueError: If at least two vectors are not provided.

    Args:
        vec_list: The list of vectors to check.

    Returns:
        `True` if `vec_list` are mutually orthogonal, and `False` otherwise.

    """
    if len(vec_list) <= 1:
        raise ValueError("There must be at least two vectors provided as input.")

    # Convert list of vectors to a 2D array (each vector is a column)
    mat = np.column_stack(vec_list)

    # Compute the matrix of inner products
    inner_product_matrix = np.dot(mat.T.conj(), mat)

    # The diagonal elements will be non-zero (norm of each vector)
    # Set the diagonal elements to zero for the comparison
    np.fill_diagonal(inner_product_matrix, 0)

    # Check if all off-diagonal elements are close to zero
    return np.allclose(inner_product_matrix, 0)
