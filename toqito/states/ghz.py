"""GHZ (Greenberger-Horne-Zeilinger) is used to represent a maximally entangled state.

In the GHZ state, the state of qubits are completely dependent on the state of other qubits. This state is an important
part of quantum computing as it is commonly used in algorithms, protocols, error corrections, cryptography, etc.
"""

import numpy as np


def ghz(dim: int, num_qubits: int, coeff: list[int] | None = None) -> np.ndarray:
    r"""Generate a (generalized) GHZ state :footcite:`Greenberger_2007_Going`.

    Returns a :code:`num_qubits`-partite GHZ state acting on :code:`dim` local dimensions, described
    in :footcite:`Greenberger_2007_Going`. For example, :code:`ghz(2, 3)` returns the standard 3-qubit GHZ state on
    qubits. The output of this function is a dense NumPy array.

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

    Using :code:`|toqito⟩`, we can see that this yields the proper state.

    .. jupyter-execute::

        from toqito.states import ghz
        ghz(2, 3)

    As this function covers the generalized GHZ state, we can consider higher dimensions. For instance here is the GHZ
    state in :math:`\mathbb{C}^{4^{\otimes 7}}` as

    .. math::
        \frac{1}{\sqrt{30}} \left(|0000000 \rangle + 2|1111111 \rangle +
        3|2222222 \rangle + 4|3333333\rangle \right).

    References
    ==========
    .. footbibliography::



    :raises ValueError: Number of qubits is not a positive integer.
    :param dim: The local dimension.
    :param num_qubits: The number of parties (qubits/qudits)
    :param coeff: (default `[1, 1, ..., 1])/sqrt(dim)`:
                  a 1-by-`dim` vector of coefficients.
    :returns: Numpy vector array as GHZ state.

    """
    if dim < 1:
        raise ValueError("InvalidDim: `dim` must be at least 1.")
    if num_qubits < 1:
        raise ValueError("InvalidNumQubits: `num_qubits` must be at least 1.")

    if coeff is None:
        coeff = np.ones(dim)
    else:
        coeff = np.array(coeff)
    if len(coeff) != dim:
        raise ValueError("InvalidCoeff: The variable `coeff` must be a vector of length equal to `dim`.")

    # Normalize coefficients if they are not.
    norm = np.linalg.norm(coeff)
    if not np.isclose(norm, 1.0):
        coeff = coeff / norm

    # Initialize the GHZ state vector.
    ghz_state = np.zeros((dim**num_qubits, 1))
    # Fill the state vector with the corresponding coefficients.
    for i in range(dim):
        # Calculate the index for the tensor product state |i, i, ..., i>.
        index = sum(i * (dim**k) for k in range(num_qubits))
        ghz_state[index] = coeff[i]

    return ghz_state
