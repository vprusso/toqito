"""Test is_unitary_antidistinguishable."""

import numpy as np
import pytest

from toqito.channel_props import is_unitary_antidistinguishable
from toqito.states import bell

PAULIS = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]


def pauli_roots(theta: float) -> list[np.ndarray]:
    """Rotations by angle theta about the three Pauli axes."""
    return [np.cos(theta / 2) * np.eye(2) + 1j * np.sin(theta / 2) * pauli for pauli in PAULIS]


@pytest.mark.parametrize(
    "unitaries, probe, expected",
    [
        # The Paulis are antidistinguishable for the optimal input strategy.
        (PAULIS, None, True),
        # ... and with the maximally entangled probe explicitly.
        (PAULIS, bell(0), True),
        # ... and even with a product probe (two output states coincide up to phase).
        (PAULIS, np.kron([1, 0], [1, 0]), True),
        # Square roots of the Paulis sit exactly at the antidistinguishability threshold.
        (pauli_roots(np.pi / 2), bell(0), True),
        # Cube roots of the Paulis are not antidistinguishable for any input strategy.
        (pauli_roots(np.pi / 3), None, False),
        # Identical unitaries can never be excluded.
        ([np.eye(2), np.eye(2)], None, False),
    ],
)
def test_is_unitary_antidistinguishable(unitaries, probe, expected):
    """Antidistinguishability of unitary collections with and without a fixed probe."""
    assert is_unitary_antidistinguishable(unitaries, probe=probe, cvxopt_kktsolver="ldl") == expected
