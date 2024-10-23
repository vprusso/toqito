"""Generalized w-state is an entangled quantum state of `n` qubits.

This state refers to the quantum superposition in which one of the qubits is in an excited state and others are in the
ground state.
"""

import numpy as np
from scipy.sparse import csr_array


def w_state(num_qubits: int, coeff: list[int] = None) -> np.ndarray:
    r"""Produce a W-state :cite:`Dur_2000_ThreeQubits`.

    Returns the W-state described in :cite:`Dur_2000_ThreeQubits`. The W-state on `num_qubits` qubits is defined by:

    .. math::
        |W \rangle = \frac{1}{\sqrt{num\_qubits}}
        \left(|100 \ldots 0 \rangle + |010 \ldots 0 \rangle + \ldots +
        |000 \ldots 1 \rangle \right).

    Examples
    ==========

    Using :code:`toqito`, we can generate the :math:`3`-qubit W-state

    .. math::
        |W_3 \rangle = \frac{1}{\sqrt{3}} \left( |100\rangle + |010 \rangle +
        |001 \rangle \right)

    as follows.

    >>> from toqito.states import w_state
    >>> w_state(3)
    array([[0.    ],
           [0.5774],
           [0.5774],
           [0.    ],
           [0.5774],
           [0.    ],
           [0.    ],
           [0.    ]])

    We may also generate a generalized :math:`W`-state. For instance, here is a :math:`4`-dimensional :math:`W`-state

    .. math::
        \frac{1}{\sqrt{30}} \left( |1000 \rangle + 2|0100 \rangle + 3|0010
        \rangle + 4 |0001 \rangle \right).

    We can generate this state in :code:`toqito` as

    >>> from toqito.states import w_state
    >>> import numpy as np
    >>> coeffs = np.array([1, 2, 3, 4]) / np.sqrt(30)
    >>> w_state(4, coeffs)
    array([[0.    ],
           [0.7303],
           [0.5477],
           [0.    ],
           [0.3651],
           [0.    ],
           [0.    ],
           [0.    ],
           [0.1826],
           [0.    ],
           [0.    ],
           [0.    ],
           [0.    ],
           [0.    ],
           [0.    ],
           [0.    ]])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: The number of qubits must be greater than or equal to 1.
    :param num_qubits: An integer representing the number of qubits.
    :param coeff: default is `[1, 1, ..., 1]/sqrt(num_qubits)`: a
                  1-by-`num_qubts` vector of coefficients.

    """
    if coeff is None:
        coeff = np.ones(num_qubits) / np.sqrt(num_qubits)

    if num_qubits < 2:
        raise ValueError("InvalidNumQubits: `num_qubits` must be at least 2.")
    if len(coeff) != num_qubits:
        raise ValueError("InvalidCoeff: The variable `coeff` must be a vector of length equal to `num_qubits`.")

    ret_w_state = csr_array((2**num_qubits, 1)).toarray()

    for i in range(num_qubits):
        ret_w_state[2**i] = coeff[num_qubits - i - 1]

    return np.around(ret_w_state, 4)
