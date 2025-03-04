"""Tests for natural_representation function."""

import numpy as np
import pytest

from toqito.channel_ops import choi_to_kraus
from toqito.channel_ops.natural_representation import natural_representation
from toqito.channels import depolarizing
from toqito.matrices import pauli
from toqito.matrix_ops import tensor

I2 = pauli("I")
X = pauli("X")
Y = pauli("Y")
Z = pauli("Z")

identity_channel = [I2]
bit_flip_channel = [np.sqrt(0.7) * I2, np.sqrt(0.3) * X]
amp_damp_channel = [np.array([[1, 0], [0, np.sqrt(0.6)]]), np.array([[0, np.sqrt(0.4)], [0, 0]])]
depol_choi = depolarizing(2, 0.1)
depol_channel = choi_to_kraus(depol_choi)


@pytest.mark.parametrize(
    "kraus_ops, expected",
    [
        # Identity channel : single Kraus operator
        (identity_channel, tensor(I2, np.conjugate(I2))),
        # Bit flip channel : two Kraus operators
        (bit_flip_channel, 0.7 * tensor(I2, np.conjugate(I2)) + 0.3 * tensor(X, np.conjugate(X))),
        # Amplitude damping channel : two Kraus operators
        (
            amp_damp_channel,
            tensor(amp_damp_channel[0], np.conjugate(amp_damp_channel[0]))
            + tensor(amp_damp_channel[1], np.conjugate(amp_damp_channel[1])),
        ),
        # Depolarizing channel : four Kraus operators
        (depol_channel, np.sum([tensor(k, np.conjugate(k)) for k in depol_channel], axis=0)),
        # Single qubit channel with different dimensions
        (
            [np.array([[1, 0, 0], [0, 1, 0]])],
            tensor(np.array([[1, 0, 0], [0, 1, 0]]), np.conjugate(np.array([[1, 0, 0], [0, 1, 0]]))),
        ),
    ],
)
def test_natural_representation_valid_inputs(kraus_ops, expected):
    """Test natural_representation function with valid inputs."""
    actual = natural_representation(kraus_ops)
    np.testing.assert_allclose(actual, expected)


def test_natural_representation_different_dimensions():
    """Test natural_representation with Kraus operators of different dimensions."""
    k1 = np.array([[1, 0], [0, 1]])
    k2 = np.array([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="All Kraus operators must have the same dimensions."):
        natural_representation([k1, k2])


def test_natural_representation_trace_preserving():
    """Test that the natural representation produces a trace-preserving map."""
    nat_rep = natural_representation(depol_channel)
    d = depol_channel[0].shape[0]
    nat_rep_reshaped = nat_rep.reshape(d**2, d**2)
    vec_identity = np.eye(d).reshape(d**2)
    result = nat_rep_reshaped @ vec_identity
    np.testing.assert_allclose(result, vec_identity, atol=1e-10)


def test_natural_representation_completeness():
    """Test completeness relation for Kraus operators via natural representation."""
    k0, k1 = amp_damp_channel
    completeness_check = np.conjugate(k0.T) @ k0 + np.conjugate(k1.T) @ k1
    np.testing.assert_allclose(completeness_check, np.eye(2), atol=1e-10)
