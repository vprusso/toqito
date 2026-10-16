"""Tests for is_quantum_channel."""

import re

import numpy as np
import pytest

from toqito.channel_ops import kraus_to_choi
from toqito.channel_props import is_quantum_channel
from toqito.channels import depolarizing


def test_is_completely_positive_kraus_false():
    """Verify non-completely positive channel as Kraus ops as False."""
    unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

    np.testing.assert_equal(is_quantum_channel(kraus_ops), False)


def test_is_completely_positive_choi_true():
    """Verify Choi matrix of the depolarizing map as a quantum channel."""
    np.testing.assert_equal(is_quantum_channel(depolarizing(2)), True)


def test_is_quantum_channel_invalid_input_returns_false():
    """Ensure is_quantum_channel returns False when given invalid input."""
    bad_input = np.eye(3)  # not a matrix of dim = perfect square.
    result = is_quantum_channel(bad_input)
    assert result is False


@pytest.mark.parametrize(
    "bad_phi",
    [
        # not ndarray or list.
        123,
        # wrong inner type.
        [["not a matrix"]],
        # mixed types.
        [[np.eye(2), "bad"]],
    ],
)
def test_is_quantum_channel_invalid_types_raise(bad_phi):
    """Testing invalid input types."""
    with pytest.raises(
        TypeError,
        match=re.escape("phi must be either a numpy array (Choi matrix) or a list of Kraus operators"),
    ):
        is_quantum_channel(bad_phi)


def test_is_quantum_channel_flat_kraus_format():
    """A flat list of Kraus operators is accepted (e.g. the identity channel)."""
    assert is_quantum_channel([np.eye(2)]) is True


# Isometric embedding of a qubit into a qutrit, V = |0><0| + |1><1|, which satisfies V^* V = I_2.
_ISOMETRY_2_TO_3 = np.array([[1, 0], [0, 1], [0, 0]])
# Kraus operators of the partial trace over the first qubit of a two-qubit system, <i| (x) I_2.
_PARTIAL_TRACE_KRAUS = [np.kron(np.identity(2)[[i]], np.identity(2)) for i in range(2)]


@pytest.mark.parametrize(
    "kraus_ops, expected_result",
    [
        # Channel from a qubit to a qutrit given by a single isometry.
        ([_ISOMETRY_2_TO_3], True),
        # Partial trace from two qubits down to one qubit.
        (_PARTIAL_TRACE_KRAUS, True),
        # Same channels in the [A, B] pair format.
        ([[_ISOMETRY_2_TO_3, _ISOMETRY_2_TO_3]], True),
        ([[k, k] for k in _PARTIAL_TRACE_KRAUS], True),
        # Completely positive but not trace preserving.
        ([_ISOMETRY_2_TO_3 / 2], False),
        ([k / 2 for k in _PARTIAL_TRACE_KRAUS], False),
    ],
)
def test_is_quantum_channel_kraus_unequal_dimensions(kraus_ops, expected_result):
    """Kraus operators of channels whose input and output dimensions differ."""
    assert is_quantum_channel(kraus_ops) is expected_result


def test_is_quantum_channel_choi_unequal_dimensions():
    """The Choi matrix of a channel with different input and output dimensions needs `dim`."""
    choi = kraus_to_choi([_ISOMETRY_2_TO_3])
    assert is_quantum_channel(choi, dim=[2, 3]) is True
    assert is_quantum_channel(choi / 2, dim=[2, 3]) is False
    # Without `dim` a 6x6 Choi matrix cannot be split into equal subsystems.
    assert is_quantum_channel(choi) is False
