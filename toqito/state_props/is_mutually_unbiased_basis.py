"""Checks if the quantum states form a mutually unbiased basis."""

from typing import Any

import numpy as np


def is_mutually_unbiased_basis(vectors: list[np.ndarray | list[float | Any]]) -> bool:
    r"""Check if list of vectors constitute a mutually unbiased basis [@WikiMUB].

    We say that two orthonormal bases

    \[
        \begin{equation}
            \mathcal{B}_0 = \left\{u_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
            \quad \text{and} \quad
            \mathcal{B}_1 = \left\{v_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
        \end{equation}
    \]

    are *mutually unbiased* if and only if
    \(\left|\langle u_a, v_b \rangle\right| = 1/\sqrt{\Sigma}\) for all \(a, b \in \Sigma\).

    For \(n \in \mathbb{N}\), a set of orthonormal bases \(\left\{
    \mathcal{B}_0, \ldots, \mathcal{B}_{n-1} \right\}\) are mutually unbiased bases if and only if
    every basis is mutually unbiased with every other basis in the set, i.e. \(\mathcal{B}_x\)
    is mutually unbiased with \(\mathcal{B}_x^{\prime}\) for all \(x \not= x^{\prime}\) with
    \(x, x^{\prime} \in \Sigma\).

    Examples:
    MUB of dimension \(2\).

    For \(d=2\), the following constitutes a mutually unbiased basis:

    \[
        \begin{equation}
            \begin{aligned}
                M_0 &= \left\{ |0 \rangle, |1 \rangle \right\}, \\
                M_1 &= \left\{ \frac{|0 \rangle + |1 \rangle}{\sqrt{2}},
                \frac{|0 \rangle - |1 \rangle}{\sqrt{2}} \right\}, \\
                M_2 &= \left\{ \frac{|0 \rangle i|1 \rangle}{\sqrt{2}},
                \frac{|0 \rangle - i|1 \rangle}{\sqrt{2}} \right\}. \\
            \end{aligned}
        \end{equation}
    \]

    ```python exec="1" source="above"
    import numpy as np
    from toqito.states import basis
    from toqito.state_props import is_mutually_unbiased_basis
    e_0, e_1 = basis(2, 0), basis(2, 1)
    mub_1 = [e_0, e_1]
    mub_2 = [1 / np.sqrt(2) * (e_0 + e_1), 1 / np.sqrt(2) * (e_0 - e_1)]
    mub_3 = [1 / np.sqrt(2) * (e_0 + 1j * e_1), 1 / np.sqrt(2) * (e_0 - 1j * e_1)]
    nested_mubs = [mub_1, mub_2, mub_3]
    mubs = sum(nested_mubs, [])
    print(is_mutually_unbiased_basis(mubs))
    ```

    Non-MUB of dimension \(2\).

    ```python exec="1" source="above"
    import numpy as np
    from toqito.states import basis
    from toqito.state_props import is_mutually_unbiased_basis
    e_0, e_1 = basis(2, 0), basis(2, 1)
    mub_1 = [e_0, e_1]
    mub_2 = [1 / np.sqrt(2) * (e_0 + e_1), e_1]
    mub_3 = [1 / np.sqrt(2) * (e_0 + 1j * e_1), e_0]
    mubs = [mub_1, mub_2, mub_3]
    print(is_mutually_unbiased_basis(mubs))
    ```

    Raises:
        ValueError: If at least two vectors are not provided.

    Args:
        vectors: The list of vectors to check.

    Returns:
        `True` if `vec_list` constitutes a mutually unbiased basis, and `False` otherwise.

    """
    num_vectors = len(vectors)
    dim_full = np.shape(vectors[0])
    dim = dim_full[0]

    # We expect the number of vectors to be a multiple of the dimension.
    if num_vectors % dim != 0:
        return False

    num_bases = num_vectors // dim

    # Check the inner product between vectors from different bases.
    for i in range(num_bases):
        for j in range(i + 1, num_bases):
            for k in range(dim):
                for litem in range(dim):
                    # Compute inner product between vectors from different bases.
                    inner_product = np.abs(np.vdot(vectors[i * dim + k], vectors[j * dim + litem])) ** 2
                    if not np.isclose(inner_product, 1 / dim):
                        return False
    return True
