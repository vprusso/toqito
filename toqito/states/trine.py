"""Trine states."""

import numpy as np

from toqito.states import basis


def trine() -> list[np.ndarray]:
    r"""Produce the set of trine states (Slide 6 of :cite:`Yard_2017_Lecture11`).

    The trine states are formally defined as:

    .. math::
        u_1 = |0\rangle, \quad
        u_1 = -\frac{1}{2}\left(|0\rangle + \sqrt{3}|1\rangle\right), \quad \text{and} \quad
        u_2 = -\frac{1}{2}\left(|0\rangle - \sqrt{3}|1\rangle\right).

    Examples
    ========

    Generating the trine states can be done by simply invoking the function:

    >>> from toqito.states import trine
    >>>
    >>> trine()
    [array([[1],
           [0]]), array([[-0.5      ],
           [-0.8660254]]), array([[-0.5      ],
           [ 0.8660254]])]


    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :return: Vector of trine states.

    """
    e_0, e_1 = basis(2, 0), basis(2, 1)
    return [
        e_0,
        -1 / 2 * (e_0 + np.sqrt(3) * e_1),
        -1 / 2 * (e_0 - np.sqrt(3) * e_1),
    ]
