"""Tests for completely_bounded_spectral_norm."""

from toqito.channel_metrics import (
    completely_bounded_spectral_norm,
    completely_bounded_trace_norm,
)
from toqito.channel_ops import dual_channel
from toqito.channels import dephasing


def test_dual_is_cb_trace_norm():
    """Test CB Spectral norm of a dephasing channel is the same as the CB Trace norm of a dephasing channel."""
    phi = dephasing(2)
    assert completely_bounded_spectral_norm(phi) == completely_bounded_trace_norm(dual_channel(phi))
