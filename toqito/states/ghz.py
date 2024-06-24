"""GHZ state."""

import numpy as np


def ghz(dim: int, num_qubits: int, coeff: list[int] | None = None) -> np.ndarray:
    r"""Generate a (generalized) GHZ state :cite:`Greenberger_2007_Going`.

    Returns a :code:`num_qubits`-partite GHZ state acting on :code:`dim` local dimensions, described
    in :cite:`Greenberger_2007_Going`. For example, :code:`ghz(2, 3)` returns the standard 3-qubit GHZ state on qubits.
    The output of this function is a dense NumPy array.

    For a system of :code:`num_qubits` qubits (i.e., :code:`dim = 2`), the GHZ state can be written
    as

    .. math::
        |GHZ \rangle = \frac{1}{\sqrt{n}} \left(|0\rangle^{\otimes n} +
        |1 \rangle^{\otimes n} \right)).

    Examples
    ==========

    When :code:`dim = 2`, and :code:`num_qubits = 3` this produces the standard GHZ state

    .. math::
        \frac{1}{\sqrt{2}} \left( |000 \rangle + |111 \rangle \right).

    Using :code:`toqito`, we can see that this yields the proper state.

    >>> from toqito.states import ghz
    >>> ghz(2, 3)
    array([[0.70710678],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.70710678]])

    As this function covers the generalized GHZ state, we can consider higher dimensions. For instance here is the GHZ
    state in :math:`\mathbb{C}^{4^{\otimes 7}}` as

    .. math::
        \frac{1}{\sqrt{30}} \left(|0000000 \rangle + 2|1111111 \rangle +
        3|2222222 \rangle + 4|3333333\rangle \right).

    Using :code:`toqito`, we can see this generates the appropriate generalized GHZ state.

    >>> from toqito.states import ghz
    >>> import numpy as np
    >>> ghz(4, 7, np.array([1, 2, 3, 4]) / np.sqrt(30))
    array([[0.18257419],
           [0.        ],
           [0.        ],
           ...,
           [0.        ],
           [0.        ],
           [0.73029674]])


    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: Number of qubits is not a positive integer.
    :param dim: The local dimension.
    :param num_qubits: The number of parties (qubits/qudits)
    :param coeff: (default `[1, 1, ..., 1])/sqrt(dim)`:
                  a 1-by-`dim` vector of coefficients.
    :returns: Numpy vector array as GHZ state.

    """
    if coeff is None:
        coeff = np.ones(dim) / np.sqrt(dim)

    # Error checking:
    if dim < 1:
        raise ValueError("InvalidDim: `dim` must be at least 1.")
    if num_qubits < 1:
        raise ValueError("InvalidNumQubits: `num_qubits` must be at least 1.")
    if len(coeff) != dim:
        raise ValueError("InvalidCoeff: The variable `coeff` must be a vector of length equal to `dim`.")

    # Initialize the GHZ state vector.
    ret_ghz_state = np.zeros((dim**num_qubits, 1))

    # Fill the GHZ state vector with the appropriate coefficients.
    for i in range(dim):
        index = sum(i * dim**k for k in range(num_qubits))
        ret_ghz_state[index] = coeff[i]

    return ret_ghz_state
