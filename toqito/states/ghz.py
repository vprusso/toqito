"""GHZ state."""
from typing import List

import numpy as np

from scipy import sparse


def ghz(dim: int, num_qubits: int, coeff: List[int] = None) -> sparse:
    r"""
    Generate a (generalized) GHZ state [GHZ07]_.

    Returns a :code:`num_qubits`-partite GHZ state acting on :code:`dim` local dimensions, described
    in [GHZ07]_. For example, :code:`ghz(2, 3)` returns the standard 3-qubit GHZ state on qubits.
    The output of this function is sparse.

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
    >>> ghz(2, 3).toarray()
    [[0.70710678],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.70710678]]

    As this function covers the generalized GHZ state, we can consider higher dimensions. For
    instance here is the GHZ state in :math:`\mathbb{C}^{4^{\otimes 7}}` as

    .. math::
        \frac{1}{\sqrt{30}} \left(|0000000 \rangle + 2|1111111 \rangle +
        3|2222222 \rangle + 4|3333333\rangle \right).

    Using :code:`toqito`, we can see this generates the appropriate generalized GHZ state.

    >>> from toqito.states import ghz
    >>> ghz(4, 7, np.array([1, 2, 3, 4]) / np.sqrt(30)).toarray()
    [[0.18257419],
     [0.        ],
     [0.        ],
     ...,
     [0.        ],
     [0.        ],
     [0.73029674]])

    References
    ==========
    .. [GHZ07] Going beyond Bell's theorem.
        D. Greenberger and M. Horne and A. Zeilinger.
        E-print: [quant-ph] arXiv:0712.0921. 2007.

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
        raise ValueError(
            "InvalidCoeff: The variable `coeff` must be a vector" " of length equal to `dim`."
        )

    # Construct the state (and do it in a way that is less memory-intensive
    # than naively tensoring things together.
    dim_sum = 1
    for i in range(1, num_qubits):
        dim_sum += dim ** i

    ret_ghz_state = sparse.lil_matrix((dim ** num_qubits, 1))
    for i in range(1, dim + 1):
        ret_ghz_state[(i - 1) * dim_sum] = coeff[i - 1]
    return ret_ghz_state
