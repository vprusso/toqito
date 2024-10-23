"""Bell states represent teh simplest examples of quantum entanglement of two qubits.

Also known as EPR pairs, Bell states comprise of four quantum states in a superposition of 0 and 1.
"""

import numpy as np

from toqito.states import basis


def bell(idx: int) -> np.ndarray:
    r"""Produce a Bell state :cite:`WikiBellSt`.

    Returns one of the following four Bell states depending on the value of :code:`idx`:

    .. math::
        \begin{equation}
            \begin{aligned}
                u_0 = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right), &
                \qquad &
                u_1 = \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right), \\
                u_2 = \frac{1}{\sqrt{2}} \left( |01 \rangle + |10 \rangle \right), &
                \qquad &
                u_3 = \frac{1}{\sqrt{2}} \left( |01 \rangle - |10 \rangle \right).
            \end{aligned}
        \end{equation}

    Examples
    ==========

    When :code:`idx = 0`, this produces the following Bell state:

    .. math::
        u_0 = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right).

    Using :code:`toqito`, we can see that this yields the proper state.

    >>> from toqito.states import bell
    >>> import numpy as np
    >>> bell(0)
    array([[0.70710678],
           [0.        ],
           [0.        ],
           [0.70710678]])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :raises ValueError: If :code:`idx` is not an integer.
    :param idx: A parameter in [0, 1, 2, 3]
    :return: Bell state with index :code:`idx`.

    """
    e_0, e_1 = basis(2, 0), basis(2, 1)
    match idx:
        case 0:
            return 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
        case 1:
            return 1 / np.sqrt(2) * (np.kron(e_0, e_0) - np.kron(e_1, e_1))
        case 2:
            return 1 / np.sqrt(2) * (np.kron(e_0, e_1) + np.kron(e_1, e_0))
        case 3:
            return 1 / np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0))
    raise ValueError("Invalid integer value for Bell state.")
