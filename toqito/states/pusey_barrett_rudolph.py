"""Construct a set of mutually unbiased bases."""

import itertools

import numpy as np

from toqito.matrices import standard_basis
from toqito.matrix_ops import tensor


def pusey_barrett_rudolph(n: int, theta: float) -> list[np.ndarray]:
    r"""Produce set of Pusey-Barrett-Rudolph (PBR) states :footcite:`Pusey_2012_On`.

    Let :math:`\theta \in [0, \pi/2]` be an angle. Define the states

    .. math::
        |\psi_0\rangle = \cos(\frac{\theta}{2})|0\rangle +
                         \sin(\frac{\theta}{2})|1\rangle
        \quad \text{and} \quad
        |\psi_1\rangle = \cos(\frac{\theta}{2})|0\rangle -
                         \sin(\frac{\theta}{2})|1\rangle.

    For some :math:`n \geq 1`, define a basis of :math:`2^n` states where

    .. math::
        |\Psi_i\rangle = |\psi_{x_i}\rangle \otimes \cdots \otimes |\psi_{x_n}\rangle.

    These PBR states are defined in Equation (A6) from :footcite:`Pusey_2012_On`.

    Examples
    ========

    Generating the PBR states can be done by simply invoking the function with a given choice of :code:`n` and
    :code:`theta`:

    .. jupyter-execute::

        from toqito.states import pusey_barrett_rudolph
        pusey_barrett_rudolph(n=1, theta=0.5)


    References
    ==========
    .. footbibliography::


    :param n: The number of states in the set.
    :param theta: Angle parameter that defines the states.
    :return: Vector of trine states.

    """
    e_0, e_1 = standard_basis(2)

    psi_0 = np.cos(theta / 2) * e_0 + np.sin(theta / 2) * e_1
    psi_1 = np.cos(theta / 2) * e_0 - np.sin(theta / 2) * e_1
    psi = [psi_0, psi_1]

    binary_strings = list(itertools.product([0, 1], repeat=n))

    states = []
    for b_str in binary_strings:
        state = []
        for b in b_str:
            state.append(psi[b])
        states.append(tensor(state))
    return states
