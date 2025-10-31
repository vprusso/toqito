"""Determine whether a collection of vectors forms a SIC POVM."""

from typing import Sequence

import numpy as np

from toqito.matrix_ops import vectors_to_gram_matrix
from toqito.state_ops import normalize


def is_sic_povm(states: Sequence[np.ndarray], *, tol: float = 1e-6) -> bool:
    r"""Check if the provided vectors yield a symmetric informationally complete POVM.

    A set of :math:`d^2` unit vectors :math:`\{\ket{\psi_j}\}` in :math:`\mathbb{C}^d` forms a
    symmetric informationally complete POVM (SIC POVM) when

    .. math::
        \left| \langle \psi_j, \psi_k \rangle \right|^2 = \frac{1}{d + 1}
        \quad \text{for all } j \neq k,

    and the projectors satisfy :math:`\sum_j \ket{\psi_j}\!\bra{\psi_j} = d \mathbb{I}`.

    Examples
    ========

    Qubit tetrahedron SIC.

    .. jupyter-execute::

        import numpy as np
        from toqito.state_props import is_sic_povm

        omega = np.exp(2j * np.pi / 3)
        sic_vectors = [
            np.array([0, 1], dtype=np.complex128),
            np.array([np.sqrt(2/3), 1/np.sqrt(3)], dtype=np.complex128),
            np.array([np.sqrt(2/3), omega / np.sqrt(3)], dtype=np.complex128),
            np.array([np.sqrt(2/3), (omega**2) / np.sqrt(3)], dtype=np.complex128),
        ]
        is_sic_povm(sic_vectors)

    Non-SIC vector family.

    .. jupyter-execute::

        import numpy as np
        from toqito.state_props import is_sic_povm
        from toqito.states import basis

        e0, e1 = basis(2, 0), basis(2, 1)
        non_sic = [e0, e1, (e0 + e1) / np.sqrt(2), (e0 - e1) / np.sqrt(2)]
        is_sic_povm(non_sic)

    References
    ==========
    .. footbibliography::

    :param states: Collection of vectors to test.
    :param tol: Numerical tolerance used for equality comparisons.
    :raises ValueError: If the vectors cannot represent valid quantum states.
    :return: :code:`True` when the vectors form a SIC POVM and :code:`False` otherwise.

    """
    if not states:
        raise ValueError("At least one vector must be provided.")

    normalized_states = [normalize(state, tol=tol) for state in states]

    dimension = normalized_states[0].size
    if any(state.size != dimension for state in normalized_states):
        raise ValueError("All SIC vectors must have the same dimension.")

    if dimension == 0:
        raise ValueError("States must have non-zero dimension.")

    num_states = len(normalized_states)
    if dimension**2 != num_states:
        return False

    gram = vectors_to_gram_matrix(normalized_states)

    if not np.allclose(np.diag(gram), 1.0, atol=tol):
        return False

    target_overlap = 1.0 / (dimension + 1.0)
    off_diag_mask = ~np.eye(num_states, dtype=bool)
    off_diag_values = np.abs(gram)**2

    if not np.allclose(off_diag_values[off_diag_mask], target_overlap, atol=tol):
        return False

    frame_operator = np.zeros((dimension, dimension), dtype=np.complex128)
    for state in normalized_states:
        frame_operator += np.outer(state, state.conj())

    if not np.allclose(frame_operator, dimension * np.eye(dimension, dtype=np.complex128), atol=tol):
        return False

    return True
