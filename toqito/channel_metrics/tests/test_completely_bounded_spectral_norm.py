"""Tests for completely_bounded_spectral_norm."""

import numpy as np

from toqito.channel_metrics import (
    completely_bounded_spectral_norm,
    completely_bounded_trace_norm,
)
from toqito.channel_ops import dual_channel, kraus_to_choi
from toqito.channels import dephasing


def test_dual_is_cb_trace_norm():
    """Test CB Spectral norm of a dephasing channel is the same as the CB Trace norm of a dephasing channel."""
    phi = dephasing(2)
    assert completely_bounded_spectral_norm(phi) == completely_bounded_trace_norm(dual_channel(phi))


def test_cb_spectral_norm_identity_channel():
    """A non-tautological check: the CB spectral norm of the identity channel is 1."""
    identity_choi = kraus_to_choi([np.eye(2)])
    assert abs(completely_bounded_spectral_norm(identity_choi) - 1.0) <= 1e-4


def test_cb_spectral_norm_rectangular_map():
    r"""Maps with unequal input/output dimensions are handled via `dim` (issue #1596).

    The qutrit-to-qubit map \(\Psi(Y) = V^\dagger Y V\), with \(V\) an isometry, is
    completely positive and unital, so its dual is a quantum channel and its completely bounded
    spectral norm is 1.
    """
    isometry = np.array([[1, 0], [0, 1], [0, 0]])
    choi = kraus_to_choi([[isometry.conj().T, isometry.conj().T]])
    assert abs(completely_bounded_spectral_norm(choi, dim=[3, 2]) - 1.0) <= 1e-3
