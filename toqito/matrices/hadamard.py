"""Generates a Hadamard matrix."""

import functools

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

        ```python exec="1" source="above" result="text"
        from toqito.matrices import hadamard

        print(hadamard(1))
        ```

    """
    if n_param < 1:
        raise ValueError("Provided parameter for matrix dimensions is invalid.")

    # H_n is the n-fold Kronecker product of the unnormalized 1-qubit Hadamard matrix, scaled
    # by 2^{-n/2}. This has entries 2^{-n/2} (-1)^{i . j} with the bitwise dot product i . j.
    h_1 = np.array([[1, 1], [1, -1]])
    return 2 ** (-n_param / 2) * functools.reduce(np.kron, [h_1] * n_param)
