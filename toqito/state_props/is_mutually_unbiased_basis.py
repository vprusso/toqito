"""Check if states form mutually unbiased basis."""
from typing import Any, List, Union

import numpy as np


def is_mutually_unbiased_basis(vec_list: List[Union[np.ndarray, List[Union[float, Any]]]]) -> bool:
    r"""
    Check if list of vectors constitute a mutually unbiased basis [WikMUB]_.

    We say that two orthonormal bases

    .. math::
        \begin{equation}
            \mathcal{B}_0 = \left\{u_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
            \quad \text{and} \quad
            \mathcal{B}_1 = \left\{v_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
        \end{equation}

    are *mutually unbiased* if and only if
    :math:`\left|\langle u_a, v_b \rangle\right| = 1/\sqrt{\Sigma}` for all :math:`a, b \in \Sigma`.

    For :math:`n \in \mathbb{N}`, a set of orthonormal bases :math:`\left\{
    \mathcal{B}_0, \ldots, \mathcal{B}_{n-1} \right\}` are mutually unbiased bases if and only if
    every basis is mutually unbiased with every other basis in the set, i.e. :math:`\mathcal{B}_x`
    is mutually unbiased with :math:`\mathcal{B}_x^{\prime}` for all :math:`x \not= x^{\prime}` with
    :math:`x, x^{\prime} \in \Sigma`.

    Examples
    ==========

    MUB of dimension :math:`2`.

    For :math:`d=2`, the following constitutes a mutually unbiased basis:

    .. math::
        \begin{equation}
            \begin{aligned}
                M_0 &= \left\{ |0 \rangle, |1 \rangle \right\}, \\
                M_1 &= \left\{ \frac{|0 \rangle + |1 \rangle}{\sqrt{2}},
                \frac{|0 \rangle - |1 \rangle}{\sqrt{2}} \right\}, \\
                M_2 &= \left\{ \frac{|0 \rangle i|1 \rangle}{\sqrt{2}},
                \frac{|0 \rangle - i|1 \rangle}{\sqrt{2}} \right\}. \\
            \end{aligned}
        \end{equation}

    >>> import numpy as np
    >>> from toqito.states import basis
    >>> from toqito.state_props import is_mutually_unbiased_basis
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> mub_1 = [e_0, e_1]
    >>> mub_2 = [1 / np.sqrt(2) * (e_0 + e_1), 1 / np.sqrt(2) * (e_0 - e_1)]
    >>> mub_3 = [1 / np.sqrt(2) * (e_0 + 1j * e_1), 1 / np.sqrt(2) * (e_0 - 1j * e_1)]
    >>> mubs = [mub_1, mub_2, mub_3]
    >>> is_mutually_unbiased_basis(mubs)
    True

    Non non-MUB of dimension :math:`2`.

    >>> import numpy as np
    >>> from toqito.states import basis
    >>> from toqito.state_props import is_mutually_unbiased_basis
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> mub_1 = [e_0, e_1]
    >>> mub_2 = [1 / np.sqrt(2) * (e_0 + e_1), e_1]
    >>> mub_3 = [1 / np.sqrt(2) * (e_0 + 1j * e_1), e_0]
    >>> mubs = [mub_1, mub_2, mub_3]
    >>> is_mutually_unbiased_basis(mubs)
    False

    References
    ==========
    .. [WikMUB] Wikipedia: Mutually unbiased bases
        https://en.wikipedia.org/wiki/Mutually_unbiased_bases

    :param vec_list: The list of vectors to check.
    :return: True if :code:`vec_list` constitutes a mutually unbiased basis, and False otherwise.
    """
    if len(vec_list) <= 1:
        raise ValueError("There must be at least two bases provided as input.")

    dim = vec_list[0][0].shape[0]
    for i, _ in enumerate(vec_list):
        for j, _ in enumerate(vec_list):
            for k in range(dim):
                if i != j:
                    if not np.isclose(
                        np.abs(np.inner(vec_list[i][k].conj().T[0], vec_list[j][k].conj().T[0]))
                        ** 2,
                        1 / dim,
                    ):
                        return False
    return True
