"""Generates a Hadamard matrix."""

import numpy as np


def hadamard(n_param: int = 1) -> np.ndarray:
    r"""Produce a `2^{n_param}` dimensional Hadamard matrix [@wikipediahadamard].

    The standard Hadamard matrix that is often used in quantum information as a
    one-qubit quantum gate is defined as

    \[
        H_1 = \frac{1}{\sqrt{2}} \begin{pmatrix}
                                    1 & 1 \\
                                    1 & -1
                                 \end{pmatrix}
    \]

    In general, the Hadamard matrix of dimension `2^{n_param}` may be
    defined as

    \[
        \left( H_n \right)_{i, j} = \frac{1}{2^{\frac{n}{2}}}
        \left(-1\right)^{i \cdot j}
    \]

    Examples:
        The standard 1-qubit Hadamard matrix can be generated in `toqito` as

        ```python exec="1" source="above"
        from toqito.matrices import hadamard

        print(hadamard(1))
        ```

    """
    if n_param < 1:
        raise ValueError("Provided parameter for matrix dimensions is invalid.")

    return 2 ** (-n_param / 2) * np.array(
        [[(-1) ** _hamming_distance(i & j) for i in range(2**n_param)] for j in range(2**n_param)]
    )


def _hamming_distance(x_param: int) -> int:
    """Calculate the bit-wise Hamming distance of `x_param` from 0.

    The Hamming distance is the number of 1s in the integer `x_param`.

    Args:
        n_param: A non-negative integer (default = 1).
        x_param: A non-negative integer.

    Returns:
        The Hamming distance of `x_param` from 0.

    """
    tot = 0
    while x_param:
        tot += 1
        x_param &= x_param - 1
    return tot
