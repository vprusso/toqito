"""Test unitary_exclusion."""

import numpy as np
import pytest

from toqito.channel_opt import unitary_exclusion
from toqito.states import bell

PAULIS = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]


def pauli_roots(theta: float) -> list[np.ndarray]:
    """Rotations by angle theta about the three Pauli axes."""
    return [np.cos(theta / 2) * np.eye(2) + 1j * np.sin(theta / 2) * pauli for pauli in PAULIS]


@pytest.mark.parametrize("primal_dual", ["primal", "dual"])
def test_paulis_antidistinguishable_with_bell_probe(primal_dual):
    """The three Pauli unitaries are perfectly excludable with a maximally entangled probe."""
    value, measurements = unitary_exclusion(PAULIS, probe=bell(0), primal_dual=primal_dual, cvxopt_kktsolver="ldl")
    assert abs(value) <= 1e-6
    assert len(measurements) == 3


def test_cube_roots_fixed_probe_value():
    """The cube roots of the Paulis with the maximally entangled probe attain a known value."""
    value, _ = unitary_exclusion(pauli_roots(np.pi / 3), probe=bell(0), cvxopt_kktsolver="ldl")
    assert abs(value - 0.0375247) <= 1e-5


def test_optimal_probe_matches_maximally_entangled_probe_for_su2_family():
    """For rotations about orthogonal axes, no probe improves on the maximally entangled one."""
    unitaries = pauli_roots(np.pi / 3)
    fixed_value, _ = unitary_exclusion(unitaries, probe=bell(0), cvxopt_kktsolver="ldl")
    optimal_value, _ = unitary_exclusion(unitaries, cvxopt_kktsolver="ldl")
    assert abs(fixed_value - optimal_value) <= 1e-5


def test_probe_without_ancilla():
    """A probe of length d is interpreted as a trivial (one-dimensional) ancilla."""
    # sigma_x|0> and sigma_y|0> are both |1> up to phase, so exclusion is perfect.
    value, _ = unitary_exclusion(PAULIS, probe=np.array([1, 0]), cvxopt_kktsolver="ldl")
    assert abs(value) <= 1e-6


def test_larger_ancilla_matches_embedded_probe():
    """Embedding the maximally entangled probe into a larger ancilla leaves the value unchanged."""
    unitaries = pauli_roots(np.pi / 3)
    probe = np.zeros(6)
    probe[0] = probe[3] = 1 / np.sqrt(2)  # |0>|0> + |1>|1> inside C^3 (x) C^2
    value_embedded, _ = unitary_exclusion(unitaries, probe=probe, cvxopt_kktsolver="ldl")
    value_qubit, _ = unitary_exclusion(unitaries, probe=bell(0), cvxopt_kktsolver="ldl")
    assert abs(value_embedded - value_qubit) <= 1e-5


def test_invalid_inputs():
    """Invalid inputs raise ValueError."""
    with pytest.raises(ValueError, match="At least 2 unitaries"):
        unitary_exclusion([PAULIS[0]])

    with pytest.raises(ValueError, match="must be unitary"):
        unitary_exclusion([PAULIS[0], np.array([[1, 1], [0, 1]], dtype=complex)])

    with pytest.raises(ValueError, match="same dimension"):
        unitary_exclusion([np.eye(2), np.eye(3)])

    with pytest.raises(ValueError, match="probe length"):
        unitary_exclusion(PAULIS, probe=np.array([1, 0, 0]))
